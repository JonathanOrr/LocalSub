import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.changes import ProposedChange
from pipeline.errors import JobCancelled
from pipeline.orchestrate import AUDIO_EXTENSIONS, PipelineConfig, redo_voice_dub, run_pipeline
from pipeline.transcript_review import validate_and_renumber

JOBS: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    video_path: str = ""
    created_at: float = field(default_factory=time.time)
    # a handful of config booleans a reconnecting browser needs to rebuild its stage
    # checklist (which stages this particular run will even attempt) without resubmitting
    # the whole form - see webapp.app.job_status and app.js's initStages
    config_flags: dict[str, bool] = field(default_factory=dict)
    status: str = "running"  # running | waiting_confirm | done | error | cancelled
    current_stage: str | None = None
    # every event ever emitted, kept so a browser that (re)connects after the run has
    # already progressed can catch up - see subscribe()
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_confirm: dict[str, Any] | None = None
    confirm_response: dict[str, Any] | None = None
    confirm_event: threading.Event = field(default_factory=threading.Event)
    # set by cancel_job() - polled cooperatively by run_pipeline (should_cancel) and by
    # _await_confirm below, since a job paused on a confirm step needs a way to be woken
    # up by cancellation too, not just by an actual browser response.
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None

    def log(self, line: str) -> None:
        self.emit({"type": "log", "line": line})

    def set_stage(self, name: str) -> None:
        self.current_stage = name
        self.emit({"type": "stage", "name": name})

    def emit(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.history.append(event)
            for q in self.subscribers:
                q.put(event)

    def subscribe(self) -> tuple[list[dict[str, Any]], queue.Queue]:
        """Atomically snapshot the job's current state and register a live queue for
        whatever happens next, so a browser connecting (or reconnecting after a page
        refresh, or switching in from the recent-jobs list) never misses or double-sees an
        event. Replaying the full "log" history is always safe (purely additive text), but
        confirm/stage/done/error events represent *state*, not a log - replaying a stale one
        (e.g. a confirm that's since been answered) would be actively wrong, so those are
        synthesized fresh from the job's current fields instead of replayed verbatim."""
        q: queue.Queue = queue.Queue()
        with self.lock:
            replay = [e for e in self.history if e.get("type") == "log"]
            if self.current_stage is not None:
                replay.append({"type": "stage", "name": self.current_stage})
            if self.pending_confirm is not None:
                replay.append({"type": "confirm_request", **self.pending_confirm})
            elif self.status == "done":
                replay.append({"type": "done", **(self.result or {})})
            elif self.status == "error":
                replay.append({"type": "error", "message": self.error})
            elif self.status == "cancelled":
                replay.append({"type": "cancelled"})
            self.subscribers.append(q)
        return replay, q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


def _changes_to_dicts(changes: list[ProposedChange]) -> list[dict]:
    return [
        {"first_cue": c.first_cue, "last_cue": c.last_cue, "summary": c.summary}
        for c in changes
    ]


def _await_confirm(job: Job, payload: dict) -> dict:
    """Shared by every make_web_confirm_* factory below: publish a confirm_request event
    onto the job (for the browser's SSE stream to pick up) and block the pipeline's
    background thread on a threading.Event until the browser POSTs its decision to
    /api/jobs/<id>/confirm (see submit_confirm), returning that decision. Each caller
    still owns its own auto_confirm short-circuit and its own interpretation of the
    response - this only covers the wait-for-browser mechanics that are otherwise
    identical across confirm kinds."""
    job.confirm_event.clear()
    job.pending_confirm = payload
    job.status = "waiting_confirm"
    job.emit({"type": "confirm_request", **payload})
    job.confirm_event.wait()
    job.pending_confirm = None
    if job.cancel_event.is_set():
        # cancel_job() also sets confirm_event to wake this exact wait - without this
        # check, a cancel clicked while paused here would just look like an empty/
        # default response and let the pipeline carry on as if nothing happened.
        raise JobCancelled("cancelled by user")
    job.status = "running"
    response = job.confirm_response or {}
    return response


def make_web_confirm_changes(job: Job):
    """Drop-in replacement for pipeline.changes.confirm_changes: same signature, but
    instead of blocking on a terminal input(), it pauses on _await_confirm."""
    def web_confirm_changes(description: str, changes: list[ProposedChange], auto_confirm: bool) -> list[ProposedChange]:
        if not changes:
            return []
        if auto_confirm:
            job.log(f"{description} (auto-confirm: applying all)")
            return changes
        response = _await_confirm(job, {
            "kind": "changes", "description": description, "changes": _changes_to_dicts(changes),
        })
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
        response = _await_confirm(job, {"kind": "primer", "primer": primer})
        if response.get("action") == "skip":
            return None
        text = (response.get("text") or "").strip()
        return text or primer
    return web_confirm_primer


def make_web_confirm_transcript(job: Job):
    """Drop-in replacement for pipeline.transcript_review.confirm_transcript: instead of
    opening $EDITOR on the file, publishes the current SRT text onto the job so the browser
    can show it in an editable textarea, then blocks until the browser POSTs the (possibly
    unedited) text back."""
    def web_confirm_transcript(srt_path: Path, auto_confirm: bool) -> None:
        if auto_confirm:
            return
        original = srt_path.read_text(encoding="utf-8")
        response = _await_confirm(job, {"kind": "transcript", "transcript": original})
        edited = response.get("text")
        if edited is None or edited == original:
            return
        srt_path.write_text(edited, encoding="utf-8")
        if not validate_and_renumber(srt_path):
            job.log("  [WARNING] edited transcript didn't look like valid SRT - reverting to the pre-edit version")
            srt_path.write_text(original, encoding="utf-8")
    return web_confirm_transcript


def make_web_confirm_primer_frames(job: Job, video_path: Path):
    """Drop-in replacement for pipeline.orchestrate.confirm_primer_frames: pauses right
    before the context primer call with the full set of frames about to be sent - both the
    automatically evenly-sampled ones and any pinned reference frames, merged - so the
    browser can review/relabel/retime/delete/add before continuing. Unlike the CLI, which
    has no interactive equivalent and just uses the default plan plus whatever
    --reference-frame flags were given. Unlabeled entries are kept (an empty label is a
    valid "just context" frame, not just a name-less reference)."""
    def web_confirm_primer_frames(
        frames: list[tuple[float, str]], auto_confirm: bool,
    ) -> list[tuple[float, str]]:
        if auto_confirm or not frames:
            return frames
        response = _await_confirm(job, {
            "kind": "primer_frames", "video_path": str(video_path),
            "frames": [{"t": t, "label": label} for t, label in frames],
        })
        edited = response.get("frames", [])
        return [(float(f["t"]), str(f.get("label", "")).strip()) for f in edited]
    return web_confirm_primer_frames


def start_job(video_path: Path, config: PipelineConfig) -> str:
    job_id = uuid.uuid4().hex[:12]
    # Mirrors run_pipeline's own is_audio_only-forces-no_llm_vision logic (see
    # pipeline/orchestrate.py) so the web UI's stage checklist shows "Vision follow-up"
    # as skipped for audio input even if the "Disable vision" checkbox wasn't checked -
    # vision is going to be skipped either way, the checklist should say so up front.
    # is_audio_only is also exposed on its own so the checklist can mark "Mux output" as
    # skipped too - run_pipeline never actually calls mux() for audio input (nothing to
    # mux subtitles into), it just fires the same stage_fn("mux") either way so the CLI/
    # web UI don't need special-cased skip handling in the pipeline itself.
    is_audio_only = video_path.suffix.lower() in AUDIO_EXTENSIONS
    job = Job(
        id=job_id, video_path=str(video_path),
        config_flags={
            "no_llm_check": config.no_llm_check, "no_context_primer": config.no_context_primer,
            "no_llm_vision": config.no_llm_vision or is_audio_only,
            "no_transcript_review": config.no_transcript_review,
            "no_translate": config.no_translate,
            "is_audio_only": is_audio_only,
            "tts_dub": config.tts_dub,
        },
    )
    JOBS[job_id] = job

    def run() -> None:
        try:
            result = run_pipeline(
                video_path, config,
                confirm_changes_fn=make_web_confirm_changes(job),
                confirm_primer_fn=make_web_confirm_primer(job),
                confirm_transcript_fn=make_web_confirm_transcript(job),
                confirm_primer_frames_fn=make_web_confirm_primer_frames(job, video_path),
                log_fn=job.log,
                stage_fn=job.set_stage,
                should_cancel=job.cancel_event.is_set,
            )
            job.result = {
                "output_path": str(result.output_path) if result.output_path else None,
                "src_srt": str(result.src_srt),
                "target_srt": str(result.target_srt) if result.target_srt else None,
                "lang": result.lang,
                "target_lang": result.target_lang,
                "video_path": str(video_path),
                "dub_audio": str(result.dub_audio) if result.dub_audio else None,
            }
            job.status = "done"
            job.emit({"type": "done", **job.result})
        except JobCancelled:
            # its own branch, not folded into the generic error path below, so the browser
            # can render this as a neutral outcome (you asked for this) rather than a failure
            job.status = "cancelled"
            job.emit({"type": "cancelled"})
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


def cancel_job(job_id: str) -> bool:
    """Ask a running (or confirm-paused) job to stop. Cooperative, not instant everywhere -
    see pipeline/errors.py:JobCancelled and run_pipeline's should_cancel parameter for why:
    whisper-cli/ffmpeg get killed outright and a confirm pause wakes immediately, but a
    genuinely in-flight LLM HTTP call finishes on its own before the next checkpoint is
    reached. Returns False if there's no job to cancel (missing, or already finished). Works
    the same for a redub-only job from start_redub_job below - both are plain Job entries in
    JOBS, cancel_job doesn't care which function is driving one."""
    job = JOBS.get(job_id)
    if job is None or job.status in ("done", "error", "cancelled"):
        return False
    job.cancel_event.set()
    job.confirm_event.set()  # wakes a paused confirm step, if any; harmless no-op otherwise
    return True


def start_redub_job(
    video_path: Path, wav_path: Path, src_srt: Path, target_srt: Path, lang: str, target_lang: str,
    out_dir: Path, is_audio_only: bool, tts_dub_model: str, tts_dub_ref_seconds: float,
    tts_dub_ref_start: float, tts_dub_ref_text: str, max_lines: int,
) -> str:
    """Re-run just the voice-clone dub (+ remux for video input) against an already-completed
    run's own transcribe/translate output, without paying whisper's/the LLM stages' cost
    again - see pipeline.orchestrate.redo_voice_dub. A plain Job like start_job's, just driven
    by a much shorter function; the web UI's "Regenerate dub" control opens its own small log/
    preview against this job's id rather than reusing the main run's stage checklist, since
    only two of its stages ever apply here."""
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, video_path=str(video_path), config_flags={"is_audio_only": is_audio_only})
    JOBS[job_id] = job

    def run() -> None:
        try:
            dub_audio, output_path = redo_voice_dub(
                video_path, wav_path, src_srt, target_srt, lang, target_lang, out_dir, is_audio_only,
                model=tts_dub_model, ref_seconds=tts_dub_ref_seconds, ref_start=tts_dub_ref_start,
                ref_text=tts_dub_ref_text, max_lines=max_lines,
                log_fn=job.log, stage_fn=job.set_stage, should_cancel=job.cancel_event.is_set,
            )
            job.result = {
                "dub_audio": str(dub_audio),
                "output_path": str(output_path) if output_path else None,
                "video_path": str(video_path),
            }
            job.status = "done"
            job.emit({"type": "done", **job.result})
        except JobCancelled:
            job.status = "cancelled"
            job.emit({"type": "cancelled"})
        except BaseException as e:
            job.status = "error"
            job.error = str(e)
            job.emit({"type": "error", "message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return job_id
