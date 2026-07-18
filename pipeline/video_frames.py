import base64
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Cap the longer side of every LLM-bound frame. Vision encoders on local models typically
# downsample/tile to a fixed native resolution well under 1080p/4K anyway, so sending frames
# at their original video resolution mostly just inflates base64 payload size and vision-token
# count (more tiles to encode) without adding anything the model can actually use - this is a
# ceiling, not a target, so frames already smaller than it pass through unscaled.
LLM_FRAME_MAX_DIM = 1024


def extract_frame_at_b64(video_path: Path, t: float) -> str | None:
    """Grab a single JPEG frame at timestamp t, base64-encoded, or None if extraction failed
    (e.g. t past the end of the video)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", str(video_path),
                "-frames:v", "1", "-vf",
                f"scale='min({LLM_FRAME_MAX_DIM},iw)':'min({LLM_FRAME_MAX_DIM},ih)':force_original_aspect_ratio=decrease",
                "-q:v", "3", str(tmp_path),
            ],
            capture_output=True,
        )
        if result.returncode == 0 and tmp_path.stat().st_size > 0:
            return base64.b64encode(tmp_path.read_bytes()).decode()
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_frames_at_b64(video_path: Path, timestamps: list[float]) -> list[str | None]:
    """Extract multiple frames in parallel - each is an independent ffmpeg subprocess/seek
    with no shared state, so there's nothing gained by doing them one at a time. Order-
    preserving (result[i] corresponds to timestamps[i]) so callers can zip them back against
    whatever metadata (label, cue) each timestamp came from."""
    if not timestamps:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(timestamps))) as pool:
        return list(pool.map(lambda t: extract_frame_at_b64(video_path, t), timestamps))


def reference_frame_content_blocks(
    video_path: Path, reference_frames: list[tuple[float, str]],
) -> list[dict]:
    """Build content blocks for a set of user-pinned reference frames (timestamp + label,
    e.g. a character's name) - each rendered as a small text label followed by the frame
    itself, so a vision-capable LLM call can ground its answer against a known face/identity
    instead of only a prose description of who's who. Shared by the context primer and the
    vision follow-up pass, so both see the same reference images."""
    if not reference_frames:
        return []
    extracted = extract_frames_at_b64(video_path, [t for t, _ in reference_frames])
    blocks: list[dict] = []
    for (_, label), frame in zip(reference_frames, extracted):
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
    timestamps = evenly_spaced_timestamps(start_s, end_s, num_frames)
    return [f for f in extract_frames_at_b64(video_path, timestamps) if f is not None]
