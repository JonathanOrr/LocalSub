import base64
import subprocess
import tempfile
from pathlib import Path


def extract_frame_at_b64(video_path: Path, t: float) -> str | None:
    """Grab a single JPEG frame at timestamp t, base64-encoded, or None if extraction failed
    (e.g. t past the end of the video)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "3", str(tmp_path),
            ],
            capture_output=True,
        )
        if result.returncode == 0 and tmp_path.stat().st_size > 0:
            return base64.b64encode(tmp_path.read_bytes()).decode()
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def reference_frame_content_blocks(
    video_path: Path, reference_frames: list[tuple[float, str]],
) -> list[dict]:
    """Build content blocks for a set of user-pinned reference frames (timestamp + label,
    e.g. a character's name) - each rendered as a small text label followed by the frame
    itself, so a vision-capable LLM call can ground its answer against a known face/identity
    instead of only a prose description of who's who. Shared by the context primer and the
    vision follow-up pass, so both see the same reference images."""
    blocks: list[dict] = []
    for t, label in reference_frames:
        frame = extract_frame_at_b64(video_path, t)
        if frame is None:
            continue
        blocks.append({"type": "text", "text": f"Reference - this is {label}:"})
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    return blocks


def evenly_spaced_timestamps(start_s: float, end_s: float, num_frames: int) -> list[float]:
    """The timestamp math extract_frames_b64 uses internally, exposed standalone so a caller
    (the context-primer frame-review confirm step) can plan/show/edit the sample points
    before any actual extraction happens."""
    span = max(end_s - start_s, 0.0)
    return [start_s + span * (i + 1) / (num_frames + 1) for i in range(num_frames)]


def extract_frames_b64(video_path: Path, start_s: float, end_s: float, num_frames: int = 3) -> list[str]:
    """Grab num_frames evenly-spaced JPEG frames between start_s and end_s, base64-encoded."""
    frames = []
    for t in evenly_spaced_timestamps(start_s, end_s, num_frames):
        frame = extract_frame_at_b64(video_path, t)
        if frame is not None:
            frames.append(frame)
    return frames
