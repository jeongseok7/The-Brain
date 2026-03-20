import json
import asyncio
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.dependencies import job_manager, neo4j_manager
from backend.config import (
    VISION_MODEL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    HIDDEN_TYPES_FILE,
    CONV_FILE,
)

router = APIRouter(tags=["System & Stats"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "vision_model": VISION_MODEL or "not configured",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@router.get("/stats")
async def get_stats():
    jobs = [
        job_manager.jobs[jid]
        for jid in job_manager.jobs_order
        if jid in job_manager.jobs
    ]
    failed = [j for j in jobs if j.status == "error"]
    active = [j for j in jobs if j.status in ("parsing", "processing")]

    total_nodes, total_relations = await neo4j_manager.get_stats()
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


@router.get("/hidden-types")
async def get_hidden_types():
    if HIDDEN_TYPES_FILE.exists():
        try:
            with open(HIDDEN_TYPES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return ["discarded", "unknown"]


@router.post("/hidden-types")
async def save_hidden_types(types: list[str]):
    with open(HIDDEN_TYPES_FILE, "w") as f:
        json.dump(types, f)
    return {"status": "ok"}


@router.get("/processed-filenames")
def processed_filenames():
    return list(job_manager.load_completed().keys())


@router.get("/jobs")
def list_jobs():
    return job_manager.get_all_jobs_dict()


@router.get("/conversations")
async def get_conversations():
    if CONV_FILE.exists():
        try:
            with open(CONV_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


@router.post("/conversations")
async def save_conversations(request: Request):
    try:
        convs = await request.json()
        with open(CONV_FILE, "w") as f:
            json.dump(convs, f)
        return {"status": "ok"}
    except Exception as e:
        logging.getLogger("uvicorn.error").error(f"Failed to save conversations: {e}")
        return {"status": "error", "detail": str(e)}


# --- REAL-TIME LOG STREAMING ---
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


sse_handler = AsyncSSELogHandler()
sse_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(sse_handler)
logging.getLogger("lightrag").addHandler(sse_handler)
logging.getLogger("raganything").addHandler(sse_handler)


@router.get("/logs/live")
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
