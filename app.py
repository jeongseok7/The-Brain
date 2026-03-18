"""
RAG-Anything FastAPI service with SSE progress streaming.

Endpoints
---------
POST /upload                - Upload & queue a document (returns job_id)
GET  /progress/{job_id}     - SSE stream of live processing events (?from_index=N)
GET  /jobs                  - List recent jobs and their status
GET  /stats                 - Aggregate stats (docs, nodes, relations, queue state)
GET  /uploads               - List files in UPLOAD_DIR with metadata
DELETE /uploads/{filename}  - Delete a file from UPLOAD_DIR
POST /queue/pause           - Pause the processing queue (current job finishes)
POST /queue/resume          - Resume the processing queue
POST /query                 - Query the knowledge graph
GET  /health                - Health check
GET  /                      - Web UI
"""

import os
import re
import uuid
import json
import shutil
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LIGHTRAG_API_URL = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

WORKING_DIR = os.getenv("WORKING_DIR", "/app/rag_storage")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
PARSER = os.getenv("PARSER", "mineru")
HIDDEN_TYPES_FILE = (
    Path(os.environ.get("WORKING_DIR", "/app/data/rag_storage")) / "hidden_types.json"
)

# Persists completed doc filenames across restarts
COMPLETED_LOG = Path(WORKING_DIR) / "completed_docs.json"


def _load_completed() -> dict:
    """Load {filename: {chunks, nodes, relations, finished_at}} from disk."""
    try:
        if COMPLETED_LOG.exists():
            return json.loads(COMPLETED_LOG.read_text())
    except Exception:
        pass
    return {}


def _save_completed(
    filename: str, chunks: int, nodes: int, relations: int, finished_at: float
):
    """Append a successfully completed doc to the persistent log."""
    try:
        docs = _load_completed()
        docs[filename] = {
            "chunks": chunks,
            "nodes": nodes,
            "relations": relations,
            "finished_at": finished_at,
        }
        COMPLETED_LOG.write_text(json.dumps(docs, indent=2))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not save completed_docs.json: {e}")


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    events: deque = field(default_factory=lambda: deque(maxlen=500))
    chunks: int = 0
    nodes: int = 0
    relations: int = 0
    error: str = ""
    block_types: dict = field(default_factory=dict)
    multimodal_progress: int = 0
    multimodal_total: int = 0

    def push(self, kind: str, message: str, **extra):
        event = {"kind": kind, "message": message, "ts": time.time(), **extra}
        self.events.append(event)
        if kind == "chunk":
            self.chunks += 1
        elif kind == "node":
            self.nodes = extra.get("total", self.nodes)
        elif kind == "relation":
            self.relations = extra.get("total", self.relations)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "chunks": self.chunks,
            "nodes": self.nodes,
            "relations": self.relations,
            "error": self.error,
            "block_types": self.block_types,
            "multimodal_progress": self.multimodal_progress,
            "multimodal_total": self.multimodal_total,
        }


_jobs: dict[str, Job] = {}
_jobs_order: deque = deque(maxlen=50)

# Queue management
_processing_queue: list[tuple[Job, str]] = []
_queue_paused: bool = False
_current_job: Job | None = None


def new_job(filename: str) -> Job:
    job = Job(id=str(uuid.uuid4()), filename=filename)
    _jobs[job.id] = job
    _jobs_order.append(job.id)
    live_ids = set(_jobs_order)
    for jid in list(_jobs.keys()):
        if jid not in live_ids:
            del _jobs[jid]
    return job


# ---------------------------------------------------------------------------
# Log handler
# ---------------------------------------------------------------------------
class JobLogHandler(logging.Handler):
    _CHUNK_PATTERNS = ("processing chunk", "chunk ", "inserting chunk", "split into")
    _NODE_PATTERNS = ("entit", "extract")
    _EDGE_PATTERNS = ("upsert_chunk",)
    _DONE_PATTERNS = ("completed merging",)

    _BLOCK_TYPE_HEADER = "content block types:"
    _BLOCK_TYPE_LINE = re.compile(r"\s*-\s*(\w+):\s*(\d+)")
    _MULTIMODAL_RE = re.compile(
        r"multimodal chunk generation progress:\s*(\d+)/(\d+)", re.IGNORECASE
    )

    def __init__(self, job: Job):
        super().__init__()
        self.job = job
        self._in_block_types = False

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        msg_lower = msg.lower()
        try:
            if self._BLOCK_TYPE_HEADER in msg_lower:
                self._in_block_types = True
                self.job.push("log", msg)
                return

            if self._in_block_types:
                m = self._BLOCK_TYPE_LINE.search(msg)
                if m:
                    btype, count = m.group(1), int(m.group(2))
                    self.job.block_types[btype] = count
                    self.job.push("block_type", msg, btype=btype, count=count)
                    return
                else:
                    self._in_block_types = False

            m = self._MULTIMODAL_RE.search(msg)
            if m:
                self.job.multimodal_progress = int(m.group(1))
                self.job.multimodal_total = int(m.group(2))
                self.job.push(
                    "multimodal_progress",
                    msg,
                    current=self.job.multimodal_progress,
                    total=self.job.multimodal_total,
                )
                return

            if any(p in msg_lower for p in self._DONE_PATTERNS):
                nums = re.findall(r"\d+", msg)
                if len(nums) >= 3:
                    self.job.nodes = int(nums[0]) + int(nums[1])
                    self.job.relations = int(nums[2])
                    self.job.push("node", msg, total=self.job.nodes)
                    self.job.push("relation", msg, total=self.job.relations)
                return

            if any(p in msg_lower for p in self._CHUNK_PATTERNS):
                self.job.push("chunk", msg)
            elif any(p in msg_lower for p in self._NODE_PATTERNS):
                nums = re.findall(r"\d+", msg)
                total = int(nums[-1]) if nums else self.job.nodes
                self.job.push("node", msg, total=total)
            elif any(p in msg_lower for p in self._EDGE_PATTERNS):
                nums = re.findall(r"\d+", msg)
                total = int(nums[-1]) if nums else self.job.relations
                self.job.push("relation", msg, total=total)
            else:
                self.job.push("log", msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reranker (BGE, CPU, loaded once at startup)
# ---------------------------------------------------------------------------
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _base_logger.info("Loading reranker BAAI/bge-reranker-v2-m3 ...")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        _base_logger.info("Reranker ready.")
    return _reranker


async def rerank_func(query: str, documents: list[str], top_n: int = 20):
    loop = asyncio.get_event_loop()
    pairs = [[query, doc] for doc in documents]
    scores = await loop.run_in_executor(None, get_reranker().predict, pairs)
    results = [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    _base_logger.info(
        f"[rerank] {len(documents)} docs → top: "
        f"{[round(r['relevance_score'], 3) for r in results[:5]]}"
    )
    return results


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
async def ollama_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    history_messages = history_messages or []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages:
        messages.append(m)
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=7200) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"num_ctx": 32768},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def ollama_vision(
    prompt,
    system_prompt=None,
    history_messages=None,
    image_data=None,
    messages=None,
    **kwargs,
):
    if not VISION_MODEL:
        return await ollama_llm(prompt, system_prompt, history_messages, **kwargs)

    if messages:
        async with httpx.AsyncClient(timeout=7200) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={"model": VISION_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    if image_data:
        built = []
        if system_prompt:
            built.append({"role": "system", "content": system_prompt})
        built.append({"role": "user", "content": prompt, "images": [image_data]})
        async with httpx.AsyncClient(timeout=7200) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": VISION_MODEL,
                    "messages": built,
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    return await ollama_llm(prompt, system_prompt, history_messages, **kwargs)


async def ollama_embed(texts: list[str]) -> np.ndarray:
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"])


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
rag: RAGAnything | None = None
_base_logger = logging.getLogger(__name__)


async def _queue_worker():
    """Processes one document at a time. Respects _queue_paused flag."""
    global _current_job
    while True:
        if not _queue_paused and _processing_queue:
            job, file_path = _processing_queue.pop(0)
            _current_job = job
            await _process_document(job, file_path)
            _current_job = None
        await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logging.basicConfig(level=logging.INFO)
    _base_logger.info("Initialising RAGAnything …")

    for d in (WORKING_DIR, UPLOAD_DIR, OUTPUT_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    for k, v in [
        ("NEO4J_URI", NEO4J_URI),
        ("NEO4J_USERNAME", NEO4J_USERNAME),
        ("NEO4J_PASSWORD", NEO4J_PASSWORD),
        ("NEO4J_DATABASE", NEO4J_DATABASE),
    ]:
        os.environ.setdefault(k, v)

    os.environ["LLM_TIMEOUT"] = os.getenv("LLM_TIMEOUT", "7200")
    os.environ["EMBEDDING_TIMEOUT"] = os.getenv("EMBEDDING_TIMEOUT", "300")

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser=PARSER,
        parse_method="auto",
        parser_output_dir=OUTPUT_DIR,
        enable_image_processing=bool(VISION_MODEL),
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    lightrag_instance = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=ollama_llm,
        llm_model_max_async=1,
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=ollama_embed,
        ),
        embedding_func_max_async=1,
        rerank_model_func=rerank_func,
    )
    await lightrag_instance.initialize_storages()

    rag = RAGAnything(
        config=config,
        lightrag=lightrag_instance,
        llm_model_func=ollama_llm,
        vision_model_func=ollama_vision if VISION_MODEL else None,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=ollama_embed,
        ),
    )

    _base_logger.info("Preloading reranker...")
    get_reranker()
    _base_logger.info("Starting queue worker...")
    asyncio.create_task(_queue_worker())
    _base_logger.info("RAGAnything ready.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
_app = FastAPI(title="RAG-Anything Service", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    _app.mount(
        "/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
    )


@_app.get("/", response_class=HTMLResponse)
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<p>Frontend not found. Place index.html in ./frontend/</p>")


_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = _app


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
async def _process_document(job: Job, file_path: str):
    assert rag is not None
    handler = JobLogHandler(job)
    handler.setLevel(logging.DEBUG)
    watched_loggers = ["lightrag", "raganything", "magic_pdf"]
    for name in watched_loggers:
        logging.getLogger(name).addHandler(handler)

    try:
        job.status = "parsing"
        job.started_at = time.time()
        job.push("status", f"Parsing {job.filename} …")
        job.push("log", "Using MinerU pipeline backend (layout detection + OCR)")
        if VISION_MODEL:
            job.push(
                "log",
                f"Vision model ({VISION_MODEL}) will process images/tables/equations",
            )

        stem = Path(file_path).stem
        for stale in Path(OUTPUT_DIR).glob(f"{stem}*"):
            if stale.is_dir():
                shutil.rmtree(stale)
                job.push("log", f"Cleared stale output directory: {stale.name}")

        await rag.process_document_complete(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            parse_method="auto",
            backend="pipeline",
            display_stats=True,
        )

        job.status = "done"
        job.finished_at = time.time()
        job.push(
            "done",
            f"✓ Finished — {job.chunks} chunks · {job.nodes} nodes · {job.relations} relations",
            chunks=job.chunks,
            nodes=job.nodes,
            relations=job.relations,
        )
        _save_completed(
            job.filename, job.chunks, job.nodes, job.relations, job.finished_at
        )

    except Exception as exc:
        job.status = "error"
        job.finished_at = time.time()
        job.error = str(exc)
        job.push("error", f"✗ {exc}")
        _base_logger.exception("Processing failed for %s", job.filename)
    finally:
        for name in watched_loggers:
            logging.getLogger(name).removeHandler(handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vision_model": VISION_MODEL or "not configured",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.get("/stats")
async def get_stats():
    jobs = [_jobs[jid] for jid in _jobs_order if jid in _jobs]
    done = [j for j in jobs if j.status == "done"]
    failed = [j for j in jobs if j.status == "error"]
    active = [j for j in jobs if j.status in ("parsing", "processing")]

    # Always query Neo4j directly — correct even after restarts
    total_nodes = total_relations = 0
    try:
        from neo4j import AsyncGraphDatabase

        async with AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        ) as driver:
            async with driver.session(database=NEO4J_DATABASE) as session:
                r1 = await session.run("MATCH (n) RETURN count(n) AS c")
                total_nodes = (await r1.single())["c"]
                r2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
                total_relations = (await r2.single())["c"]
    except Exception as e:
        _base_logger.warning(f"Neo4j stats query failed: {e}")

    # Count processed from persistent log — survives restarts
    completed = _load_completed()
    processed_count = len(completed)

    return {
        "total_docs": processed_count,
        "processed": processed_count,
        "failed": len(failed),
        "active": len(active),
        "queued": len(_processing_queue),
        "total_nodes": total_nodes,
        "total_relations": total_relations,
        "queue_paused": _queue_paused,
        "current_job": _current_job.to_dict() if _current_job else None,
    }


@_app.get("/hidden-types")
async def get_hidden_types():
    if HIDDEN_TYPES_FILE.exists():
        try:
            with open(HIDDEN_TYPES_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    # Default hidden types for brand new users
    return ["discarded", "unknown"]


@_app.post("/hidden-types")
async def save_hidden_types(types: list[str]):
    with open(HIDDEN_TYPES_FILE, "w") as f:
        json.dump(types, f)
    return {"status": "ok"}


@app.get("/processed-filenames")
def processed_filenames():
    """Returns filenames that completed successfully — survives restarts."""
    return list(_load_completed().keys())


@app.get("/graph")
async def get_graph(limit: int = 300, search: str = ""):
    """
    Return graph data for 3D visualization.
    - No search: top N nodes by connection count
    - With search: 2-hop neighborhood around matching nodes
    """
    from neo4j import AsyncGraphDatabase

    try:
        async with AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        ) as driver:
            async with driver.session(database=NEO4J_DATABASE) as session:
                if search.strip():
                    # Neighborhood search — find matching nodes then expand 2 hops
                    cypher = """
                        MATCH (n)
                        WHERE toLower(n.entity_id) CONTAINS toLower($search)
                           OR toLower(coalesce(n.description,'')) CONTAINS toLower($search)
                        WITH n LIMIT 5
                        MATCH path = (n)-[r*0..2]-(neighbor)
                        WITH collect(DISTINCT startNode(relationships(path)[0])) +
                             collect(DISTINCT endNode(relationships(path)[0])) +
                             collect(DISTINCT n) AS allNodes,
                             collect(DISTINCT r) AS allRels
                        UNWIND allNodes AS node
                        WITH collect(DISTINCT node)[..200] AS topNodes
                        MATCH (a)-[r]->(b)
                        WHERE a IN topNodes AND b IN topNodes
                        RETURN
                          a.entity_id AS src, a.entity_type AS src_type,
                          coalesce(a.description,'') AS src_desc,
                          b.entity_id AS tgt, b.entity_type AS tgt_type,
                          coalesce(b.description,'') AS tgt_desc,
                          coalesce(r.description, type(r)) AS rel_label,
                          coalesce(r.weight, 1.0) AS weight
                        LIMIT 2000
                    """
                    result = await session.run(cypher, search=search.strip())
                else:
                    # Top N nodes by degree, then edges between them
                    cypher = """
                        MATCH (n)
                        WITH n, size([(n)--() | 1]) AS degree
                        ORDER BY degree DESC
                        LIMIT $limit
                        WITH collect(n) AS topNodes
                        MATCH (a)-[r]->(b)
                        WHERE a IN topNodes AND b IN topNodes
                        RETURN
                          a.entity_id AS src, a.entity_type AS src_type,
                          coalesce(a.description,'') AS src_desc,
                          b.entity_id AS tgt, b.entity_type AS tgt_type,
                          coalesce(b.description,'') AS tgt_desc,
                          coalesce(r.description, type(r)) AS rel_label,
                          coalesce(r.weight, 1.0) AS weight
                        LIMIT 3000
                    """
                    result = await session.run(cypher, limit=limit)

                nodes: dict[str, dict] = {}
                links: list[dict] = []

                async for row in result:
                    src, tgt = row["src"], row["tgt"]
                    if not src or not tgt:
                        continue

                    if src not in nodes:
                        nodes[src] = {
                            "id": src,
                            "type": (row["src_type"] or "unknown").lower(),
                            "desc": row["src_desc"][:200] if row["src_desc"] else "",
                            "degree": 0,
                        }
                    if tgt not in nodes:
                        nodes[tgt] = {
                            "id": tgt,
                            "type": (row["tgt_type"] or "unknown").lower(),
                            "desc": row["tgt_desc"][:200] if row["tgt_desc"] else "",
                            "degree": 0,
                        }

                    nodes[src]["degree"] += 1
                    nodes[tgt]["degree"] += 1
                    links.append(
                        {
                            "source": src,
                            "target": tgt,
                            "label": (row["rel_label"] or "")[:80],
                            "weight": float(row["weight"] or 1.0),
                        }
                    )

                return {"nodes": list(nodes.values()), "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lightrag/documents")
async def lightrag_documents():
    """Proxy to LightRAG GET /documents — returns all docs grouped by status."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{LIGHTRAG_API_URL}/documents")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LightRAG API not reachable")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/lightrag/pipeline_status")
async def lightrag_pipeline_status():
    """Proxy to LightRAG GET /documents/pipeline_status."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{LIGHTRAG_API_URL}/documents/pipeline_status")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LightRAG API not reachable")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/jobs")
def list_jobs():
    return [_jobs[jid].to_dict() for jid in reversed(list(_jobs_order)) if jid in _jobs]


@app.get("/uploads")
def list_uploads():
    files = []
    for f in Path(UPLOAD_DIR).iterdir():
        if f.is_file():
            files.append(
                {
                    "filename": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return sorted(files, key=lambda x: x["modified"], reverse=True)


@app.delete("/uploads/{filename}")
def delete_upload(filename: str):
    safe = Path(UPLOAD_DIR) / Path(filename).name
    if not safe.exists():
        raise HTTPException(status_code=404, detail="File not found")
    safe.unlink()
    return {"deleted": filename}


@app.post("/queue/pause")
def pause_queue():
    global _queue_paused
    _queue_paused = True
    return {"paused": True}


@app.post("/queue/resume")
def resume_queue():
    global _queue_paused
    _queue_paused = False
    return {"paused": False}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    dest = Path(UPLOAD_DIR) / file.filename
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = new_job(file.filename)
    job.push("queued", f"File received: {file.filename}")
    _processing_queue.append((job, str(dest)))

    return {
        "job_id": job.id,
        "filename": file.filename,
        "queue_position": len(_processing_queue),
    }


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str, from_index: int = 0):
    """SSE stream. ?from_index=N resumes without replaying already-seen events."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate() -> AsyncGenerator[str, None]:
        job = _jobs[job_id]
        sent = from_index
        idle = 0
        while True:
            events = list(job.events)
            while sent < len(events):
                yield f"data: {json.dumps({'index': sent, **events[sent]})}\n\n"
                sent += 1
                idle = 0

            if job.status in ("done", "error"):
                events = list(job.events)
                while sent < len(events):
                    yield f"data: {json.dumps({'index': sent, **events[sent]})}\n\n"
                    sent += 1
                break

            idle += 1
            if idle % 50 == 0:
                yield ": keepalive\n\n"

            await asyncio.sleep(0.3)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class QueryRequest(BaseModel):
    question: str
    mode: str = "mix"
    only_need_context: bool = False
    return_nodes: bool = False


# Intercept LightRAG query logs to capture retrieved entity names
class QueryLogCapture(logging.Handler):
    """Captures the 'Query nodes:' log line emitted by LightRAG during a query."""

    _QUERY_NODES_RE = re.compile(
        r"Query nodes?:\s*(.+?)(?:\s*\(top_k|$)", re.IGNORECASE
    )

    def __init__(self):
        super().__init__()
        self.entity_names: list[str] = []

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        m = self._QUERY_NODES_RE.search(msg)
        if m:
            raw = m.group(1)
            self.entity_names = [e.strip() for e in raw.split(",") if e.strip()]


async def _resolve_node_ids(entity_names: list[str]) -> list[str]:
    """Look up entity_ids in Neo4j that match the captured entity names."""
    if not entity_names:
        return []
    try:
        from neo4j import AsyncGraphDatabase

        async with AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        ) as driver:
            async with driver.session(database=NEO4J_DATABASE) as session:
                # Match by exact entity_id or case-insensitive partial match
                result = await session.run(
                    """
                    UNWIND $names AS name
                    MATCH (n)
                    WHERE toLower(n.entity_id) = toLower(name)
                       OR toLower(n.entity_id) CONTAINS toLower(name)
                    RETURN DISTINCT n.entity_id AS eid
                    LIMIT 60
                    """,
                    names=entity_names,
                )
                ids = []
                async for row in result:
                    if row["eid"]:
                        ids.append(row["eid"])
                return ids
    except Exception as e:
        _base_logger.warning(f"Node ID resolution failed: {e}")
        return []


@app.post("/query")
async def query(req: QueryRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")

    # Attach log capture handler if caller wants highlighted nodes
    capture = QueryLogCapture() if req.return_nodes else None
    if capture:
        for name in ["lightrag", "raganything"]:
            logging.getLogger(name).addHandler(capture)

    try:
        answer = await rag.aquery(req.question, mode=req.mode, vlm_enhanced=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if capture:
            for name in ["lightrag", "raganything"]:
                logging.getLogger(name).removeHandler(capture)

    if req.only_need_context:
        if not answer:
            return {"context": "No relevant information found.", "mode": req.mode}
        return {"context": str(answer), "mode": req.mode}

    result: dict = {
        "answer": str(answer) if answer else "No results found.",
        "mode": req.mode,
    }

    if req.return_nodes and capture:
        result["highlighted_nodes"] = await _resolve_node_ids(capture.entity_names)

    return result
