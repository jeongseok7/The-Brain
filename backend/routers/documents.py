import re
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from backend.dependencies import job_manager, state
from backend.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES

router = APIRouter(tags=["Documents & Queue"])


@router.get("/uploads")
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


@router.delete("/uploads/{filename}")
def delete_upload(filename: str):
    safe = Path(UPLOAD_DIR) / Path(filename).name
    if not safe.exists():
        raise HTTPException(status_code=404, detail="File not found")
    safe.unlink()
    return {"deleted": filename}


@router.post("/queue/pause")
def pause_queue():
    job_manager.queue_paused = True
    return {"paused": True}


@router.post("/queue/resume")
def resume_queue():
    job_manager.queue_paused = False
    return {"paused": False}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if state.rag is None:
        raise HTTPException(status_code=503, detail="RAGAnything not initialised yet")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")

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


@router.get("/progress/{job_id}")
async def progress_stream(job_id: str, from_index: int = 0):
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
