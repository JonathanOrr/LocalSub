import base64
import subprocess
import tempfile
from pathlib import Path


def extract_frames_b64(video_path: Path, start_s: float, end_s: float, num_frames: int = 3) -> list[str]:
    """Grab num_frames evenly-spaced JPEG frames between start_s and end_s, base64-encoded."""
    span = max(end_s - start_s, 0.0)
    frames = []
    for i in range(num_frames):
        frac = (i + 1) / (num_frames + 1)
        t = start_s + span * frac
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "3", str(tmp_path),
                ],
                capture_output=True,
            )
            if result.returncode == 0 and tmp_path.stat().st_size > 0:
                frames.append(base64.b64encode(tmp_path.read_bytes()).decode())
        finally:
            tmp_path.unlink(missing_ok=True)
    return frames
