import subprocess
import sys
from pathlib import Path

# repo root (parent of this pipeline/ package) - whisper.cpp/ lives alongside subtranslate.py
SCRIPT_DIR = Path(__file__).resolve().parent.parent
WHISPER_DIR = SCRIPT_DIR / "whisper.cpp"
WHISPER_CLI = WHISPER_DIR / "build" / "bin" / "whisper-cli"
MODELS_DIR = WHISPER_DIR / "models"
VAD_MODEL_NAME = "silero-v6.2.0"

# whisper language code -> (ISO 639-2 tag for mkv metadata, display title)
LANGUAGE_INFO = {
    "ja": ("jpn", "Japanese"),
    "zh": ("chi", "Chinese"),
    "en": ("eng", "English"),
    "ko": ("kor", "Korean"),
    "es": ("spa", "Spanish"),
    "fr": ("fre", "French"),
    "de": ("ger", "German"),
}


def language_info(lang: str) -> tuple[str, str]:
    return LANGUAGE_INFO.get(lang, (lang, lang.title()))


def check_dependencies(model: str, use_vad: bool, vad_engine: str) -> tuple[Path, Path | None]:
    if not WHISPER_CLI.exists():
        sys.exit(f"whisper-cli not found at {WHISPER_CLI} - build whisper.cpp first")
    model_path = MODELS_DIR / f"ggml-{model}.bin"
    if not model_path.exists():
        sys.exit(
            f"Model not found at {model_path} - download it with:\n"
            f"  cd {WHISPER_DIR} && bash models/download-ggml-model.sh {model}"
        )
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        sys.exit("ffmpeg not found on PATH")

    vad_model_path = None
    if use_vad and vad_engine == "whisper":
        vad_model_path = MODELS_DIR / f"ggml-{VAD_MODEL_NAME}.bin"
        if not vad_model_path.exists():
            sys.exit(
                f"VAD model not found at {vad_model_path} - download it with:\n"
                f"  cd {WHISPER_DIR} && bash models/download-vad-model.sh {VAD_MODEL_NAME}\n"
                f"or drop --vad to run without it (now the default)"
            )
    elif use_vad and vad_engine == "ten":
        try:
            import ten_vad  # noqa: F401
        except ImportError:
            sys.exit(
                "ten-vad not installed - run: pip install ten-vad\n"
                "(also needs the system libc++ runtime, e.g. `sudo dnf install libcxx` on Fedora)"
            )
    return model_path, vad_model_path


def extract_audio(video_path: Path, workdir: Path) -> Path:
    wav_path = workdir / f"{video_path.stem}.wav"
    print(f"[1/4] Extracting audio -> {wav_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
    )
    return wav_path


def transcribe(
    wav_path: Path, model_path: Path, lang: str, translate: bool,
    out_stem: Path, threads: int, use_gpu: bool, vad_model_path: Path | None,
    vad_max_speech_s: float, entropy_thold: float, logprob_thold: float, no_speech_thold: float,
    max_context: int, vad_threshold: float, vad_min_speech_ms: int, vad_min_silence_ms: int,
    vad_speech_pad_ms: int,
) -> Path:
    cmd = [
        str(WHISPER_CLI), "-m", str(model_path), "-f", str(wav_path),
        "-l", lang, "-osrt", "-of", str(out_stem), "-np", "-t", str(threads),
        "-et", str(entropy_thold), "-lpt", str(logprob_thold), "-nth", str(no_speech_thold),
        "-mc", str(max_context),
    ]
    if translate:
        cmd.append("-tr")
    if not use_gpu:
        cmd.append("--no-gpu")
    if vad_model_path is not None:
        cmd += [
            "--vad", "-vm", str(vad_model_path), "-vmsd", str(vad_max_speech_s),
            "-vt", str(vad_threshold), "-vspd", str(vad_min_speech_ms),
            "-vsd", str(vad_min_silence_ms), "-vp", str(vad_speech_pad_ms),
        ]
    subprocess.run(cmd, check=True)
    return Path(f"{out_stem}.srt")


def mux(
    video_path: Path, src_srt: Path, src_lang: str, target_srt: Path | None, target_lang: str,
    output_path: Path,
) -> None:
    print(f"[mux] Muxing subtitles -> {output_path}")
    src_tag, src_title = language_info(src_lang)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-i", str(src_srt)]
    if target_srt is not None:
        cmd += ["-i", str(target_srt), "-map", "0:v", "-map", "0:a", "-map", "1:0", "-map", "2:0"]
    else:
        cmd += ["-map", "0:v", "-map", "0:a", "-map", "1:0"]
    cmd += [
        "-c:v", "copy", "-c:a", "copy", "-c:s", "srt",
        "-metadata:s:s:0", f"language={src_tag}", "-metadata:s:s:0", f"title={src_title}",
    ]
    if target_srt is not None:
        target_tag, target_title = language_info(target_lang)
        cmd += ["-metadata:s:s:1", f"language={target_tag}", "-metadata:s:s:1", f"title={target_title}"]
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)
