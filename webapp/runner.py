import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.changes import ProposedChange
from pipeline.orchestrate import PipelineConfig, run_pipeline

JOBS: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    status: str = "running"  # running | waiting_confirm | done | error
    events: queue.Queue = field(default_factory=queue.Queue)
    pending_confirm: dict[str, Any] | None = None
    confirm_response: dict[str, Any] | None = None
    confirm_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None

    def log(self, line: str) -> None:
        self.events.put({"type": "log", "line": line})

    def emit(self, event: dict[str, Any]) -> None:
        self.events.put(event)


def _changes_to_dicts(changes: list[ProposedChange]) -> list[dict]:
    return [
        {"first_cue": c.first_cue, "last_cue": c.last_cue, "summary": c.summary}
        for c in changes
    ]


def make_web_confirm_changes(job: Job):
    """Drop-in replacement for pipeline.changes.confirm_changes: same signature, but
    instead of blocking on a terminal input(), it publishes the proposed changes onto the
    job (for the browser's SSE stream to pick up) and blocks on a threading.Event until the
    browser POSTs its decision to /api/jobs/<id>/confirm."""
    def web_confirm_changes(description: str, changes: list[ProposedChange], auto_confirm: bool) -> list[ProposedChange]:
        if not changes:
            return []
        if auto_confirm:
            job.log(f"{description} (auto-confirm: applying all)")
            return changes
        job.confirm_event.clear()
        job.pending_confirm = {
            "kind": "changes", "description": description, "changes": _changes_to_dicts(changes),
        }
        job.status = "waiting_confirm"
        job.emit({"type": "confirm_request", **job.pending_confirm})
        job.confirm_event.wait()
        job.status = "running"
        response = job.confirm_response or {}
        job.pending_confirm = None
        selected = set(response.get("selected", range(len(changes))))
        return [c for i, c in enumerate(changes) if i in selected]
    return web_confirm_changes


def make_web_confirm_primer(job: Job):
    """Drop-in replacement for pipeline.context_primer.confirm_context_primer, using the
    same wait-for-browser-decision mechanism as make_web_confirm_changes."""
    def web_confirm_primer(primer: str, auto_confirm: bool) -> str | None:
        if auto_confirm:
            job.log("Context primer generated (auto-confirm: using as-is)")
            return primer
        job.confirm_event.clear()
        job.pending_confirm = {"kind": "primer", "primer": primer}
        job.status = "waiting_confirm"
        job.emit({"type": "confirm_request", **job.pending_confirm})
        job.confirm_event.wait()
        job.status = "running"
        response = job.confirm_response or {}
        job.pending_confirm = None
        if response.get("action") == "skip":
            return None
        text = (response.get("text") or "").strip()
        return text or primer
    return web_confirm_primer


def start_job(video_path: Path, config: PipelineConfig) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id)
    JOBS[job_id] = job

    def run() -> None:
        try:
            result = run_pipeline(
                video_path, config,
                confirm_changes_fn=make_web_confirm_changes(job),
                confirm_primer_fn=make_web_confirm_primer(job),
                log_fn=job.log,
            )
            job.result = {
                "output_path": str(result.output_path),
                "src_srt": str(result.src_srt),
                "target_srt": str(result.target_srt) if result.target_srt else None,
                "lang": result.lang,
                "target_lang": result.target_lang,
            }
            job.status = "done"
            job.emit({"type": "done", **job.result})
        except BaseException as e:
            job.status = "error"
            job.error = str(e)
            job.emit({"type": "error", "message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return job_id


def submit_confirm(job_id: str, response: dict[str, Any]) -> bool:
    job = JOBS.get(job_id)
    if job is None or job.pending_confirm is None:
        return False
    job.confirm_response = response
    job.confirm_event.set()
    return True
