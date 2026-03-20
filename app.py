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
import json
import shutil
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# LightRAG & RAG-Anything
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from raganything import RAGAnything, RAGAnythingConfig

# Local imports
from backend.llm_providers import OllamaProvider, OpenAIProvider
from backend.jobs import JobManager, JobLogHandler, Job
from backend.neo4j_utils import Neo4jManager
from backend.reranker import DocumentReranker
from backend.config import *
from backend.dependencies import (
    job_manager,
    neo4j_manager,
    document_reranker,
    state,
    base_logger,
)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


async def _queue_worker():
    """Processes documents. Respects queue_paused flag."""
    while True:
        if not job_manager.queue_paused and job_manager.processing_queue:
            job, file_path = job_manager.processing_queue.popleft()
            job_manager.current_job = job
            await _process_document(job, file_path)
            job_manager.current_job = None
        await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    base_logger.info("Initialising RAGAnything …")

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

    # === AUTO-SWITCH LOGIC ===
    if OPENAI_API_KEY:
        base_logger.info("OPENAI_API_KEY detected. Routing LLM to External API.")
        provider = OpenAIProvider(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            llm_model=LLM_MODEL,
            vision_model=VISION_MODEL,
            embed_model=EMBEDDING_MODEL,
            timeout=LLM_TIMEOUT,
        )
    else:
        base_logger.info("No OPENAI_API_KEY detected. Routing LLM to local Ollama.")
        provider = OllamaProvider(
            base_url=OLLAMA_BASE_URL,
            llm_model=LLM_MODEL,
            vision_model=VISION_MODEL,
            embed_model=EMBEDDING_MODEL,
            num_ctx=LLM_NUM_CTX,
            timeout=LLM_TIMEOUT,
        )

    lightrag_instance = LightRAG(
        working_dir=WORKING_DIR,
        graph_storage="Neo4JStorage",
        llm_model_func=provider.llm,
        llm_model_max_async=LLM_MAX_ASYNC,
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=provider.embed,
        ),
        embedding_func_max_async=EMBEDDING_MAX_ASYNC,
        rerank_model_func=document_reranker.rerank,
    )
    await lightrag_instance.initialize_storages()

    state.rag = RAGAnything(
        config=config,
        lightrag=lightrag_instance,
        llm_model_func=provider.llm,
        vision_model_func=provider.vision if VISION_MODEL else None,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=provider.embed,
        ),
    )

    base_logger.info("Preloading reranker...")
    document_reranker.load()
    base_logger.info("Starting queue worker...")
    asyncio.create_task(_queue_worker())
    base_logger.info("RAGAnything ready.")
    neo4j_manager.connect()
    yield
    await neo4j_manager.close()


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
    if state.rag is None:
        job.status = "error"
        job.push("error", "RAGAnything not initialised")
        return

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

        await state.rag.process_document_complete(
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
        job_manager.save_completed(
            job.filename, job.chunks, job.nodes, job.relations, job.finished_at
        )

    except Exception as exc:
        job.status = "error"
        job.finished_at = time.time()
        job.error = str(exc)
        job.push("error", f"✗ {exc}")
        base_logger.exception("Processing failed for %s", job.filename)
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
    jobs = [
        job_manager.jobs[jid]
        for jid in job_manager.jobs_order
        if jid in job_manager.jobs
    ]
    failed = [j for j in jobs if j.status == "error"]
    active = [j for j in jobs if j.status in ("parsing", "processing")]

    # Always query Neo4j directly — correct even after restarts
    total_nodes, total_relations = await neo4j_manager.get_stats()

    # Count processed from persistent log — survives restarts
    completed = job_manager.load_completed()
    processed_count = len(completed)

    return {
        "total_docs": processed_count,
        "processed": processed_count,
        "failed": len(failed),
        "active": len(active),
        "queued": len(job_manager.processing_queue),
        "total_nodes": total_nodes,
        "total_relations": total_relations,
        "queue_paused": job_manager.queue_paused,
        "current_job": job_manager.current_job.to_dict()
        if job_manager.current_job
        else None,
    }


@_app.get("/hidden-types")
async def get_hidden_types():
    if HIDDEN_TYPES_FILE.exists():
        try:
            with open(HIDDEN_TYPES_FILE, "r") as f:
                return json.load(f)
        except Exception:
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
    return list(job_manager.load_completed().keys())


@app.get("/graph")
async def get_graph(limit: int = 300, search: str = ""):
    """
    Return graph data for 3D visualization.
    - No search: top N nodes by connection count
    - With search: 2-hop neighborhood around matching nodes
    """
    try:
        return await neo4j_manager.get_graph(limit, search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
def list_jobs():
    return job_manager.get_all_jobs_dict()


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
    job_manager.queue_paused = True
    return {"paused": True}


@app.post("/queue/resume")
def resume_queue():
    job_manager.queue_paused = False
    return {"paused": False}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if state.rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
    # Validate extension
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")

    # Validate size (read into memory, check, then write)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 500 MB)")

    dest = Path(UPLOAD_DIR) / file.filename
    with dest.open("wb") as fh:
        fh.write(content)

    job = job_manager.new_job(file.filename)
    job.push("queued", f"File received: {file.filename}")
    job_manager.processing_queue.append((job, str(dest)))

    return {
        "job_id": job.id,
        "filename": file.filename,
        "queue_position": len(job_manager.processing_queue),
    }


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str, from_index: int = 0):
    """SSE stream. ?from_index=N resumes without replaying already-seen events."""
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate() -> AsyncGenerator[str, None]:
        job = job_manager.jobs[job_id]
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


@app.post("/query")
async def query(req: QueryRequest):
    if state.rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")

    # Attach log capture handler if caller wants highlighted nodes
    capture = QueryLogCapture() if req.return_nodes else None
    if capture:
        for name in ["lightrag", "raganything"]:
            logging.getLogger(name).addHandler(capture)

    try:
        answer = await state.rag.aquery(req.question, mode=req.mode, vlm_enhanced=False)
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
        result["highlighted_nodes"] = await neo4j_manager.resolve_node_ids(
            capture.entity_names
        )

    return result


@_app.get("/conversations")
async def get_conversations():
    if CONV_FILE.exists():
        try:
            with open(CONV_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@_app.post("/conversations")
async def save_conversations(request: Request):
    try:
        # Grab the raw JSON directly to avoid FastAPI validation errors
        convs = await request.json()
        with open(CONV_FILE, "w") as f:
            json.dump(convs, f)
        return {"status": "ok"}
    except Exception as e:
        base_logger.error(f"Failed to save conversations: {e}")
        return {"status": "error", "detail": str(e)}


# --- 2. REAL-TIME LOG STREAMING ---
live_log_clients = set()


class AsyncSSELogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        level = record.levelname.lower()
        data = json.dumps({"level": level, "message": msg})

        try:
            loop = asyncio.get_running_loop()
            for q in list(live_log_clients):
                loop.call_soon_threadsafe(self._safe_put, q, data)
        except RuntimeError:
            pass

    @staticmethod
    def _safe_put(q, data):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# Attach to root logger
sse_handler = AsyncSSELogHandler()
sse_handler.setFormatter(logging.Formatter("%(message)s"))

logging.getLogger().addHandler(sse_handler)
logging.getLogger("lightrag").addHandler(sse_handler)
logging.getLogger("raganything").addHandler(sse_handler)


@_app.get("/logs/live")
async def live_logs_stream(request: Request):
    q = asyncio.Queue(maxsize=200)
    live_log_clients.add(q)

    async def log_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"
        except Exception:
            pass
        finally:
            if q in live_log_clients:
                live_log_clients.remove(q)

    return StreamingResponse(log_generator(), media_type="text/event-stream")
