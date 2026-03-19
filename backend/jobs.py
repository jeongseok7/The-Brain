import time
import uuid
import json
import re
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
# Job structure
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


# Logging
class JobLogHandler(logging.Handler):
    """Parses LightRAG/MinerU logs to track node, relation, and chunk progress."""

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


class JobManager:
    """Manages all document queues, active jobs, and persistent disk tracking."""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.completed_log = self.working_dir / "completed_docs.json"

        # State variables that used to be globals
        self.jobs: dict[str, Job] = {}
        self.jobs_order: deque = deque(maxlen=50)
        self.processing_queue: deque[tuple[Job, str]] = deque()
        self.queue_paused: bool = False
        self.current_job: Job | None = None

    def load_completed(self) -> dict:
        """Load {filename: {chunks, nodes, relations, finished_at}} from disk."""
        try:
            if self.completed_log.exists():
                return json.loads(self.completed_log.read_text())
        except Exception:
            pass
        return {}

    def save_completed(
        self, filename: str, chunks: int, nodes: int, relations: int, finished_at: float
    ):
        """Append a successfully completed doc to the persistent log."""
        try:
            docs = self.load_completed()
            docs[filename] = {
                "chunks": chunks,
                "nodes": nodes,
                "relations": relations,
                "finished_at": finished_at,
            }
            self.completed_log.write_text(json.dumps(docs, indent=2))
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Could not save completed_docs.json: {e}"
            )

    def new_job(self, filename: str) -> Job:
        """Create a new job and push it to the tracking system."""
        job = Job(id=str(uuid.uuid4()), filename=filename)
        self.jobs[job.id] = job
        self.jobs_order.append(job.id)

        # Keep dict memory clean by only keeping what's in the deque
        live_ids = set(self.jobs_order)
        for jid in list(self.jobs.keys()):
            if jid not in live_ids:
                del self.jobs[jid]
        return job

    def get_all_jobs_dict(self) -> list[dict]:
        """Return a formatted list of jobs for the API endpoint."""
        return [
            self.jobs[jid].to_dict()
            for jid in reversed(list(self.jobs_order))
            if jid in self.jobs
        ]
