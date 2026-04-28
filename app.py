import os
import time
import shutil
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# LightRAG & RAG-Anything
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

# Local backend modules
from backend.llm_providers import OllamaProvider, OpenAIProvider
from backend.dependencies import (
    job_manager,
    neo4j_manager,
    document_reranker,
    state,
    base_logger,
)
from backend.jobs import Job, JobLogHandler
from backend.config import *

# Import our new API routers
from backend.routers import documents, graph, system


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


async def _queue_worker():
    """Processes documents. Respects queue_paused flag."""
    while True:
        if not job_manager.queue_paused and job_manager.processing_queue:
            job, file_path = job_manager.processing_queue.popleft()
            job_manager.current_job = job
            await _process_document(job, file_path)
            job_manager.current_job = None
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
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

    os.environ["LLM_TIMEOUT"] = str(LLM_TIMEOUT)
    os.environ["EMBEDDING_TIMEOUT"] = str(EMBEDDING_TIMEOUT)

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser=PARSER,
        parse_method="auto",
        parser_output_dir=OUTPUT_DIR,
        enable_image_processing=bool(VISION_MODEL),
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # LLM engine routing
    # Settings for every provider
    common_kwargs = {
        "llm_model": LLM_MODEL,
        "vision_model": VISION_MODEL,
        "embed_model": EMBEDDING_MODEL,
        "timeout": LLM_TIMEOUT,
    }

    # Map engines to their specific Provider class and unique variables
    engine_registry = {
        "openai": {
            "class": OpenAIProvider,
            "kwargs": {"api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL},
        },
        "vllm": {
            "class": OpenAIProvider,
            "kwargs": {"api_key": OPENAI_API_KEY, "base_url": VLLM_BASE_URL},
        },
        "lmstudio": {
            "class": OpenAIProvider,
            "kwargs": {"api_key": OPENAI_API_KEY, "base_url": LM_STUDIO_BASE_URL},
        },
        "llamacpp": {
            "class": OpenAIProvider,
            "kwargs": {"api_key": OPENAI_API_KEY, "base_url": LLAMA_CPP_BASE_URL},
        },
        "ollama": {
            "class": OllamaProvider,
            "kwargs": {"base_url": OLLAMA_BASE_URL, "num_ctx": LLM_NUM_CTX},
        },
    }

    # Look up the chosen engine (fallback to 'ollama')
    selected_engine = engine_registry.get(LLM_ENGINE, engine_registry["ollama"])
    ProviderClass = selected_engine["class"]
    specific_kwargs = selected_engine["kwargs"]

    # Instantiate the provider
    provider = ProviderClass(**common_kwargs, **specific_kwargs)

    base_logger.info(
        f"Routing LLM using '{LLM_ENGINE}' engine via {ProviderClass.__name__} "
        f"at {specific_kwargs.get('base_url')}"
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

    base_logger.info("Starting queue worker...")
    asyncio.create_task(_queue_worker())
    base_logger.info("RAGAnything ready.")
    neo4j_manager.connect()
    yield
    await neo4j_manager.close()


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG-Anything Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plug in the routers
app.include_router(documents.router)
app.include_router(graph.router)
app.include_router(system.router)

# Mount Frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount(
        "/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
    )


@app.get("/", response_class=HTMLResponse)
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<p>Frontend not found. Place index.html in ./frontend/</p>")
