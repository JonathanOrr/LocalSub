import select
import subprocess
import sys
from pathlib import Path
from typing import Callable

from pipeline.errors import JobCancelled

# How often should_cancel() gets polled while a subprocess is otherwise silent - see
# _run_streamed. whisper-cli in particular can go silent for several seconds at a time
# (model load, then however long a single chunk takes to decode) with zero output lines in
# between, so checking cancellation only between lines isn't actually responsive - it would
# only take effect whenever the next line happens to arrive, which could be a long wait.
CANCEL_POLL_S = 0.3

# repo root (parent of this pipeline/ package) - whisper.cpp/ lives alongside localsub.py
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


# Every source language whisper.cpp can transcribe (its g_lang table in src/whisper.cpp,
# same commit setup.sh pins) - used to populate the web UI's source-language dropdown so it
# can't be pointed at a code whisper.cpp doesn't actually recognize.
WHISPER_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
}


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


def _run_streamed(
    cmd: list[str], log_fn: Callable[[str], None], should_cancel: Callable[[], bool] = lambda: False,
) -> None:
    """Run cmd, forwarding its stdout/stderr line-by-line to log_fn as it runs (instead of
    subprocess.run's default of just inheriting the parent's stdio) - needed so a caller
    like the web UI can capture live progress instead of it going straight to a terminal
    the caller doesn't own. Raises subprocess.CalledProcessError on nonzero exit, matching
    subprocess.run(..., check=True). should_cancel is polled on a fixed timer (CANCEL_POLL_S)
    via select(), not just between output lines - whisper-cli in particular can run silent
    for several real seconds (model load, then however long one chunk takes to decode), and
    checking only between lines would leave a cancel request sitting unnoticed for that whole
    stretch. On a hit, the process is killed outright and JobCancelled is raised, giving the
    web UI's Cancel button a genuinely prompt effect regardless of the subprocess's own
    output cadence."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None  # guaranteed by stdout=subprocess.PIPE above
    stdout = proc.stdout
    while True:
        if should_cancel():
            proc.kill()
            proc.wait()
            raise JobCancelled("cancelled by user")
        ready, _, _ = select.select([stdout], [], [], CANCEL_POLL_S)
        if ready:
            line = stdout.readline()
            if line == "":
                break  # EOF - the process is finishing up
            log_fn(line.rstrip("\n"))
        elif proc.poll() is not None:
            break  # exited with nothing left to read
    proc.wait()
    if should_cancel():
        raise JobCancelled("cancelled by user")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def extract_audio(
    video_path: Path, workdir: Path, log_fn: Callable[[str], None] = print,
    should_cancel: Callable[[], bool] = lambda: False,
) -> Path:
    wav_path = workdir / f"{video_path.stem}.wav"
    log_fn(f"[1/4] Extracting audio -> {wav_path}")
    _run_streamed(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
        ],
        log_fn, should_cancel,
    )
    return wav_path


def transcribe(
    wav_path: Path, model_path: Path, lang: str, translate: bool,
    out_stem: Path, threads: int, use_gpu: bool, vad_model_path: Path | None,
    vad_max_speech_s: float, entropy_thold: float, logprob_thold: float, no_speech_thold: float,
    max_context: int, vad_threshold: float, vad_min_speech_ms: int, vad_min_silence_ms: int,
    vad_speech_pad_ms: int, log_fn: Callable[[str], None] = print,
    should_cancel: Callable[[], bool] = lambda: False,
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
    _run_streamed(cmd, log_fn, should_cancel)
    return Path(f"{out_stem}.srt")


def mux(
    video_path: Path, src_srt: Path, src_lang: str, target_srt: Path | None, target_lang: str,
    output_path: Path, log_fn: Callable[[str], None] = print,
    should_cancel: Callable[[], bool] = lambda: False,
) -> None:
    log_fn(f"[mux] Muxing subtitles -> {output_path}")
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
    _run_streamed(cmd, log_fn, should_cancel)
