import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.whisper_engine import SCRIPT_DIR, run_streamed

# The venv an earlier session set up specifically for this feature (ROCm torch + qwen-tts -
# see amd_instructions/ and pytorch_instructions/), kept separate from the system Python
# webui.py/localsub.py run under. Heavy, GPU-vendor-specific ML deps have no business being
# a hard dependency of the whole project just to transcribe/translate - see
# pipeline/tts_worker.py's module docstring.
TTS_VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"

# Language codes qwen3-tts's -Base voice-clone models support, mapped to the display name
# the model itself expects (not the 2-letter code used everywhere else in this pipeline).
# Voice dubbing is silently skipped (with a log line, not a hard failure) for any
# target_lang outside this set - see run_pipeline in orchestrate.py.
TTS_LANGS = {
    "en": "English", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


@dataclass
class DubResult:
    full_wav: Path
    lines_dir: Path


def check_tts_dependencies() -> None:
    if not TTS_VENV_PYTHON.exists():
        sys.exit(
            f"TTS venv not found at {TTS_VENV_PYTHON} - voice-clone dubbing needs its own "
            f"venv with torch + qwen-tts installed (see amd_instructions/ and "
            f"pytorch_instructions/ for the ROCm setup used on this machine; a CUDA or "
            f"CPU torch build works the same way on other hardware). Uncheck 'Voice-clone "
            f"dub' / drop --tts-dub to skip this stage."
        )


def generate_voice_dub(
    wav_path: Path, src_srt: Path, target_srt: Path, target_lang: str, video_stem: str,
    out_dir: Path, model: str, ref_seconds: float, ref_start: float = 0.0, ref_text: str = "",
    max_lines: int = 0,
    log_fn: Callable[[str], None] = print, should_cancel: Callable[[], bool] = lambda: False,
) -> DubResult:
    """Clone the speaker's voice from wav_path and speak every line of target_srt in it,
    placed at that line's own cue start time so the result lines up with the source video's
    timeline - see pipeline/tts_worker.py (run as a subprocess under TTS_VENV_PYTHON) for the
    actual generation. ref_start picks where in wav_path the reference clip is sliced from
    (default: the very beginning); ref_text, when non-empty, overrides the default of
    auto-deriving the reference transcript from src_srt with a hand-typed ground-truth one;
    max_lines caps how many target_srt lines are generated (0 = all) - for quickly auditioning
    a reference clip/text/model choice, see webapp's "Regenerate dub" control. Raises
    subprocess.CalledProcessError on failure (e.g. insufficient VRAM - the worker prints why
    and exits nonzero before that)."""
    check_tts_dependencies()
    lang_name = TTS_LANGS[target_lang]
    dub_dir = out_dir / f"{video_stem}.{target_lang}.dub"
    full_wav = dub_dir / f"{video_stem}.{target_lang}.dub.wav"
    cmd = [
        str(TTS_VENV_PYTHON), "-m", "pipeline.tts_worker",
        "--wav", str(wav_path), "--ref-srt", str(src_srt), "--target-srt", str(target_srt),
        "--lang-name", lang_name, "--out-dir", str(dub_dir), "--out-wav", str(full_wav),
        "--model", model, "--ref-seconds", str(ref_seconds), "--ref-start", str(ref_start),
    ]
    if ref_text.strip():
        cmd += ["--ref-text", ref_text]
    if max_lines:
        cmd += ["--max-lines", str(max_lines)]
    # cwd=SCRIPT_DIR so `-m pipeline.tts_worker` resolves the pipeline package from the repo
    # root under the *other* venv's interpreter, same as this process's own sys.path.
    run_streamed(cmd, log_fn, should_cancel, cwd=SCRIPT_DIR)
    return DubResult(full_wav=full_wav, lines_dir=dub_dir / "lines")
