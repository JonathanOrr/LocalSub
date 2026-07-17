#!/usr/bin/env python3
import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import NamedTuple

SRT_TIMESTAMP_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
LLM_CHUNK_SIZE = 40
LLM_TRANSLATE_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")
LLM_NOTE_CUE_RE = re.compile(r"^\[(\d+)(?:-(\d+))?\]")


def notes_for_batch(notes: list[str], batch: list[tuple[str, str, str]]) -> list[str]:
    batch_nums = {int(num) for num, ts, text in batch}
    relevant = []
    for note in notes:
        m = LLM_NOTE_CUE_RE.match(note)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if any(lo <= n <= hi for n in batch_nums):
            relevant.append(note)
    return relevant

SCRIPT_DIR = Path(__file__).resolve().parent
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


def srt_timestamp_range_to_seconds(ts_range: str) -> tuple[float, float]:
    m = SRT_TIMESTAMP_RE.search(ts_range)
    if m is None:
        raise ValueError(f"not a valid SRT timestamp range: {ts_range!r}")
    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
    start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
    end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    return start, end


def seconds_to_srt_timestamp(s: float) -> str:
    total_ms = round(max(s, 0.0) * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


TEN_VAD_HOP_SIZE = 256  # 16ms frames at 16kHz


def ten_vad_speech_segments(
    wav_path: Path, threshold: float, min_speech_ms: int, min_silence_ms: int, pad_ms: int,
    max_speech_s: float,
) -> list[tuple[float, float]]:
    """Run TEN VAD (https://github.com/TEN-framework/ten-vad) over a 16kHz mono 16-bit PCM
    wav and return merged/padded speech segments as (start_s, end_s) in the wav's own
    timeline. Used as a pre-pass replacement for whisper.cpp's built-in Silero VAD - TEN VAD
    detects speech/non-speech transitions faster and catches short silences between
    back-to-back sentences that Silero can miss."""
    try:
        from ten_vad import TenVad
    except ImportError:
        sys.exit(
            "ten-vad not installed - run: pip install ten-vad\n"
            "(also needs the system libc++ runtime, e.g. `sudo dnf install libcxx` on Fedora)"
        )
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            sys.exit(f"ten-vad expects 16kHz mono 16-bit PCM, got a differently-shaped wav: {wav_path}")
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    sr = 16000
    frame_s = TEN_VAD_HOP_SIZE / sr
    vad = TenVad(TEN_VAD_HOP_SIZE, threshold)
    num_frames = len(samples) // TEN_VAD_HOP_SIZE
    flags = [False] * num_frames
    for i in range(num_frames):
        frame = samples[i * TEN_VAD_HOP_SIZE : (i + 1) * TEN_VAD_HOP_SIZE]
        _, flag = vad.process(frame)
        flags[i] = bool(flag)

    # raw speech runs, as [first_frame, last_frame] index pairs
    runs = []
    i = 0
    while i < num_frames:
        if flags[i]:
            j = i
            while j + 1 < num_frames and flags[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    if not runs:
        return []

    # merge runs separated by a silence gap shorter than min_silence_ms
    min_silence_frames = (min_silence_ms / 1000) / frame_s
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] - 1 <= min_silence_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # discard runs shorter than min_speech_ms (noise blips, not real speech)
    min_speech_frames = (min_speech_ms / 1000) / frame_s
    merged = [(a, b) for a, b in merged if (b - a + 1) >= min_speech_frames]
    if not merged:
        return []

    # convert to seconds, pad, then re-merge any segments padding brought back together.
    # total_s uses the true sample count (not num_frames * frame_s) so padding can still
    # reach into the leftover partial frame at the very end instead of truncating it away.
    total_s = len(samples) / sr
    pad_s = pad_ms / 1000
    padded = [[max(0.0, a * frame_s - pad_s), min(total_s, (b + 1) * frame_s + pad_s)] for a, b in merged]
    segments = [padded[0]]
    for start_s, end_s in padded[1:]:
        if start_s <= segments[-1][1]:
            segments[-1][1] = max(segments[-1][1], end_s)
        else:
            segments.append([start_s, end_s])

    # force-split any segment longer than max_speech_s - bounds repetition-loop risk on long
    # unbroken speech, same rationale as whisper.cpp's own --vad-max-speech-s
    final = []
    for start_s, end_s in segments:
        dur = end_s - start_s
        if max_speech_s <= 0 or dur <= max_speech_s:
            final.append((start_s, end_s))
            continue
        n_chunks = math.ceil(dur / max_speech_s)
        chunk_len = dur / n_chunks
        final.extend(
            (start_s + k * chunk_len, min(end_s, start_s + (k + 1) * chunk_len))
            for k in range(n_chunks)
        )
    return final


def build_vad_trimmed_wav(
    wav_path: Path, segments: list[tuple[float, float]], out_path: Path, gap_ms: int = 350,
) -> list[tuple[float, float, float]]:
    """Concatenate only the given speech segments from wav_path into out_path, with gap_ms
    of digital silence inserted between them - separate from the anti-clipping padding
    already baked into each segment, this gives whisper.cpp's own internal segmenter an
    unambiguous silence to split subtitle cues on, so distinct back-to-back sentences don't
    get merged into one cue just because the real gap between them was thin. Returns a trim
    map of (trimmed_start_s, trimmed_end_s, orig_start_s) per segment, used afterward to
    remap transcription timestamps (produced against the trimmed audio) back onto the
    original timeline."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.int16)
    trim_map = []
    chunks = []
    cursor_s = 0.0
    for idx, (start_s, end_s) in enumerate(segments):
        if idx > 0:
            chunks.append(gap)
            cursor_s += len(gap) / sr
        start_i, end_i = int(start_s * sr), int(end_s * sr)
        chunks.append(samples[start_i:end_i])
        dur_s = (end_i - start_i) / sr
        trim_map.append((cursor_s, cursor_s + dur_s, start_s))
        cursor_s += dur_s

    trimmed = np.concatenate(chunks) if chunks else samples[:0]
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(trimmed.tobytes())
    return trim_map


def remap_time(t: float, trim_map: list[tuple[float, float, float]]) -> float:
    for trimmed_start, trimmed_end, orig_start in trim_map:
        if t <= trimmed_end:
            return orig_start + max(0.0, t - trimmed_start)
    trimmed_start, trimmed_end, orig_start = trim_map[-1]
    return orig_start + (trimmed_end - trimmed_start)


def remap_srt_timestamps(srt_path: Path, trim_map: list[tuple[float, float, float]]) -> None:
    """Rewrite an SRT's timestamps from the VAD-trimmed audio timeline back onto the
    original video timeline, using the trim map from build_vad_trimmed_wav."""
    cues = parse_srt_cues(srt_path)
    remapped = []
    for num, ts, text in cues:
        start_s, end_s = srt_timestamp_range_to_seconds(ts)
        new_ts = (
            f"{seconds_to_srt_timestamp(remap_time(start_s, trim_map))} --> "
            f"{seconds_to_srt_timestamp(remap_time(end_s, trim_map))}"
        )
        remapped.append((num, new_ts, text))
    write_srt(remapped, srt_path)


class RepeatFlag(NamedTuple):
    first_cue: str
    last_cue: str
    text: str
    count: int
    start_ts: str
    end_ts: str
    # set only for intra-cue character-run flags (detect_char_repeats): the exact repeated
    # substring within `text` that needs collapsing, e.g. 'うううううううううううう'.
    # None for cue-level flags (detect_repeats), where `text` itself is the repeated unit.
    char_run: str | None = None


class ProposedChange(NamedTuple):
    first_cue: str
    last_cue: str
    summary: str
    # new (num placeholder, timestamp, text) cues to splice in; None = delete the range entirely
    replacement: list[tuple[str, str, str]] | None


class RationalityFlag(NamedTuple):
    first_cue: str
    last_cue: str
    issue: str
    needs_vision: bool
    proposed_fix: str | None


def parse_srt_cues(srt_path: Path) -> list[tuple[str, str, str]]:
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()
    cues = []
    for b in content.strip().split("\n\n"):
        lines = b.strip().split("\n")
        if len(lines) < 3:
            continue
        cues.append((lines[0], lines[1], " ".join(lines[2:]).strip()))
    return cues


def detect_repeats(srt_path: Path, repeat_threshold: int) -> list[RepeatFlag]:
    """Flag runs of repeat_threshold+ consecutive identical cues - a hallucination-loop
    signature. Purely mathematical (exact string match), no LLM involved."""
    cues = parse_srt_cues(srt_path)
    flags = []
    i = 0
    while i < len(cues):
        j = i
        while j + 1 < len(cues) and cues[j + 1][2] == cues[i][2]:
            j += 1
        count = j - i + 1
        if count >= repeat_threshold:
            flags.append(RepeatFlag(
                first_cue=cues[i][0], last_cue=cues[j][0], text=cues[i][2], count=count,
                start_ts=cues[i][1].split("-->")[0].strip(),
                end_ts=cues[j][1].split("-->")[1].strip(),
            ))
        i = j + 1
    return flags


CHAR_REPEAT_RE = re.compile(r"(\S)\1{3,}")


def detect_char_repeats(cues: list[tuple[str, str, str]], repeat_threshold: int) -> list[RepeatFlag]:
    """Flag cues whose text itself contains the same character repeated repeat_threshold+
    times in a row - the intra-cue analogue of detect_repeats' consecutive-identical-cue
    check. Catches hallucinated strings like 'うううううううううううう' that live inside a
    single cue rather than spanning several duplicated ones. Purely mathematical (regex
    run-length match), no LLM involved."""
    flags = []
    for num, ts, text in cues:
        longest = None
        for m in CHAR_REPEAT_RE.finditer(text):
            if longest is None or len(m.group(0)) > len(longest.group(0)):
                longest = m
        if longest is not None and len(longest.group(0)) >= repeat_threshold:
            start_ts, end_ts = ts.split("-->")
            flags.append(RepeatFlag(
                first_cue=num, last_cue=num, text=text, count=len(longest.group(0)),
                start_ts=start_ts.strip(), end_ts=end_ts.strip(), char_run=longest.group(0),
            ))
    return flags


def format_sent_content(sent: str | list[dict]) -> str:
    if isinstance(sent, str):
        return f"```\n{sent}\n```"
    parts = []
    frame_num = 0
    for block in sent:
        if block.get("type") == "text":
            parts.append(f"```\n{block['text']}\n```")
        elif block.get("type") == "image_url":
            frame_num += 1
            url = block["image_url"]["url"]
            parts.append(f"Frame {frame_num}:\n\n![frame {frame_num}]({url})")
    return "\n\n".join(parts)


def write_raw_log_entry(
    raw_log_path: Path, first_cue: str, last_cue: str, sent: str | list[dict], message: dict,
) -> None:
    with open(raw_log_path, "a", encoding="utf-8") as f:
        f.write(f"## Cues {first_cue}-{last_cue}\n\n")
        f.write(f"### Sent\n\n{format_sent_content(sent)}\n\n")
        f.write(f"### Reasoning\n\n{message.get('reasoning_content') or '*(none)*'}\n\n")
        f.write(f"### Output\n\n```\n{message.get('content') or '(empty)'}\n```\n\n")
        f.write("---\n\n")


def call_llm(
    endpoint: str, model: str, content: str | list[dict], frequency_penalty: float | None = 0.4,
    presence_penalty: float | None = 0.3, temperature: float = 1.0, top_p: float | None = 0.95,
    top_k: int | None = 64, max_tokens: int = 20000, timeout: int = 600,
) -> dict:
    """POST a chat completion request, return the response message dict. Raises
    urllib.error.URLError / OSError / TimeoutError on connection failure.

    temperature/top_p/top_k default to Google's recommended sampling config for Gemma
    (https://ai.google.dev/gemma/docs/core/model_card_4) rather than greedy decoding
    (temperature=0) - greedy decoding is more prone to the kind of deterministic
    repetition/oscillation loops we hit repeatedly with this model. frequency_penalty and
    presence_penalty are kept on top as an extra safety net, not a replacement.
    Trade-off: no longer fully deterministic run-to-run."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    req = urllib.request.Request(
        f"{endpoint}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]


def write_srt(cues: list[tuple[str, str, str]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for num, ts, text in cues:
            f.write(f"{num}\n{ts}\n{text}\n\n")


LLM_REPEAT_PROMPT = (
    "The groups below each show a transcript passage where an automatic transcription "
    "repeated the same content an implausible number of times in a row - a known "
    "hallucination-loop failure mode, not real speech. This shows up two ways: either the "
    "exact same subtitle cue repeated back-to-back across multiple cues, or the same "
    "character/syllable repeated many times within a single cue's text (e.g. "
    "'うううううううううううう'). For each group, decide what to do:\n"
    "- keep_one: the repeated content is plausible, just looped by mistake - collapse to a single instance\n"
    "- keep_n=N: some repetition is genuinely plausible (e.g. rhythmic exclamations, or a "
    "held/elongated sound) - N is a small number well under the original count\n"
    "- replace: the repeated content itself looks wrong/garbled - give corrected replacement "
    "text for the whole cue\n"
    "Reply with EXACTLY one line per group below, in the same order, and nothing else - no "
    "explanation before, after, or between lines. Use each group's own cue-number range in "
    "place of first-last. Format:\n"
    "[first-last] keep_one\n"
    "[first-last] keep_n=2\n"
    "[first-last] replace: corrected text here\n\n"
)
LLM_REPEAT_LINE_RE = re.compile(
    r"^\[(\d+)-(\d+)\]\s*(keep_one|keep_n=(\d+)|replace)\s*:?\s*(.*)$", re.IGNORECASE,
)


def llm_resolve_repeats(
    cues: list[tuple[str, str, str]], repeats: list[RepeatFlag], endpoint: str, model: str,
    raw_log_path: Path | None = None,
) -> list[ProposedChange]:
    """Ask the LLM what to do with each flagged repeat-loop group. Sends only the flagged
    snippets (not full chunks), so this stays cheap regardless of file length."""
    if not repeats:
        return []
    cue_by_num = {int(num): ts for num, ts, text in cues}
    lines = [f"[{r.first_cue}-{r.last_cue}] repeated {r.count}x: {r.text!r}" for r in repeats]
    sent_content = LLM_REPEAT_PROMPT + "\n".join(lines)
    try:
        message = call_llm(endpoint, model, sent_content)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  [WARNING] repeat-resolution LLM call failed: {e}")
        return []
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM repeat-resolution log\n\n")
        write_raw_log_entry(raw_log_path, repeats[0].first_cue, repeats[-1].last_cue, sent_content, message)
    reply = (message.get("content") or "").strip()

    decisions = {}
    for line in reply.splitlines():
        m = LLM_REPEAT_LINE_RE.match(line.strip())
        if m:
            decisions[(m.group(1), m.group(2))] = m

    changes = []
    for r in repeats:
        m = decisions.get((r.first_cue, r.last_cue))
        if m is None:
            continue
        action = m.group(3).lower()
        first_ts = cue_by_num.get(int(r.first_cue), r.start_ts)
        if action == "keep_one" and r.char_run is not None:
            new_text = r.text.replace(r.char_run, r.char_run[0], 1)
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated character run collapsed to 1x -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
        elif action == "keep_one":
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x -> collapse to 1x: {r.text!r}",
                [(r.first_cue, first_ts, r.text)],
            ))
        elif action.startswith("keep_n") and r.char_run is not None:
            n = max(1, min(int(m.group(4) or 1), r.count))
            new_text = r.text.replace(r.char_run, r.char_run[0] * n, 1)
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated character run collapsed to {n}x -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
        elif action.startswith("keep_n"):
            n = max(1, min(int(m.group(4) or 1), r.count))
            keep_nums = [int(r.first_cue) + k for k in range(n)]
            replacement = [(str(k), cue_by_num.get(k, first_ts), r.text) for k in keep_nums]
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x -> collapse to {n}x: {r.text!r}", replacement,
            ))
        elif action == "replace":
            new_text = m.group(5).strip() or r.text
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x, text likely wrong -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
    return changes


def confirm_changes(description: str, changes: list[ProposedChange], auto_confirm: bool) -> list[ProposedChange]:
    if not changes:
        return []
    print(f"\n{description}")
    for i, c in enumerate(changes, 1):
        print(f"  {i}. cues {c.first_cue}-{c.last_cue}: {c.summary}")
    if auto_confirm:
        print("  (--auto-confirm: applying all)")
        return changes
    response = input("Apply all? [Y/n], or list numbers to exclude (e.g. 2,5): ").strip()
    if response == "" or response.lower() in ("y", "yes"):
        return changes
    if response.lower() in ("n", "no"):
        return []
    exclude = {int(x) for x in re.findall(r"\d+", response)}
    return [c for i, c in enumerate(changes, 1) if i not in exclude]


def apply_changes(
    cues: list[tuple[str, str, str]], changes: list[ProposedChange],
) -> list[tuple[str, str, str]]:
    """Splice confirmed changes into the cue list (matched by original cue number) and
    renumber sequentially. A change's replacement=None deletes its whole range; otherwise
    the range is replaced by the given (placeholder_num, ts, text) tuples."""
    change_by_first = {int(c.first_cue): c for c in changes}
    nums = [int(num) for num, ts, text in cues]
    num_to_idx = {n: idx for idx, n in enumerate(nums)}

    result: list[tuple[str, str]] = []
    i = 0
    while i < len(cues):
        change = change_by_first.get(nums[i])
        if change is None:
            result.append((cues[i][1], cues[i][2]))
            i += 1
            continue
        if change.replacement is not None:
            result.extend((ts, text) for _, ts, text in change.replacement)
        i = num_to_idx.get(int(change.last_cue), i) + 1
    return [(str(n), ts, text) for n, (ts, text) in enumerate(result, start=1)]


LLM_CONTEXT_PRIMER_PROMPT = (
    "You are given the full transcript of a video's spoken dialogue, along with a handful "
    "of frames sampled evenly across its runtime. Write a short primer to help with later "
    "line-by-line review: who the speakers/characters seem to be (names, roles, "
    "relationships if inferable), the setting, the tone/genre, and the general throughline "
    "of what's happening. This is a best-effort inference from limited context, not a "
    "verified summary - keep it to a few short paragraphs, and don't invent specifics "
    "you're not reasonably confident in. Reply with ONLY the primer text, and nothing else "
    "- no commentary before or after."
)


def build_context_primer(
    cues: list[tuple[str, str, str]], video_path: Path, num_frames: int, endpoint: str, model: str,
    raw_log_path: Path | None = None,
) -> str | None:
    """One-time pass over the whole (cleaned) transcript plus a fixed number of frames
    sampled evenly across the video, producing a short context primer (characters, setting,
    tone, throughline) that gets prepended to every later rationality-check and translation
    call. Cost stays flat regardless of video length: one call, a bounded frame count, and a
    short requested output - not a per-chunk or per-frame cost."""
    if not cues:
        return None
    transcript_text = "\n".join(f"[{num}] {text}" for num, ts, text in cues)
    _, duration_s = srt_timestamp_range_to_seconds(cues[-1][1])

    content: list[dict] = [{
        "type": "text",
        "text": f"{LLM_CONTEXT_PRIMER_PROMPT}\n\nTranscript:\n{transcript_text}",
    }]
    frames = extract_frames_b64(video_path, 0.0, duration_s, num_frames) if num_frames > 0 else []
    for b64 in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    try:
        message = call_llm(endpoint, model, content, max_tokens=2000)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  [WARNING] context-primer generation failed: {e}")
        return None
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM context-primer log\n\n")
        write_raw_log_entry(raw_log_path, cues[0][0], cues[-1][0], content, message)
    primer = (message.get("content") or "").strip()
    if not primer:
        print("  [WARNING] context-primer produced no output")
        return None
    return primer


def confirm_context_primer(primer: str, auto_confirm: bool) -> str | None:
    print("\nContext primer (characters/setting/tone/throughline) inferred from the transcript and sampled frames:\n")
    print(primer)
    if auto_confirm:
        print("\n  (--auto-confirm: using as-is)")
        return primer
    response = input(
        "\nUse this as context for the rest of the pipeline? [Y/n], or 'e' to edit, or 'skip' to discard: "
    ).strip().lower()
    if response in ("", "y", "yes"):
        return primer
    if response in ("n", "no", "skip"):
        return None
    if response in ("e", "edit"):
        print("Enter the corrected primer, then an empty line to finish:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        edited = "\n".join(lines).strip()
        return edited or primer
    return primer


LLM_RATIONALITY_PROMPT = (
    "You are reviewing an automatic speech-to-text transcript for problems beyond simple "
    "repetition (already handled separately): garbled/grammatically broken text, or a "
    "phrase that seems fabricated/out of place given the surrounding context (e.g. a "
    "plausible-sounding but likely mistranscribed line, or a probable homophone/mis-hearing). "
    "For each cue you flag, decide whether you're confident enough from the text alone to "
    "propose a fix, or whether you'd need to see the video at that moment to judge it "
    "properly. Always give your best-guess fix either way.\n"
    "Reply with ONLY a list of flagged cues, one per line, and nothing else - no explanation "
    "before, after, or between lines. Keep ISSUE to a few words and never use the | character "
    "inside ISSUE or FIX, since it separates the fields below. Format:\n"
    "[N] NEEDS_VISION: yes|no | ISSUE: brief reason | FIX: your best-guess corrected text\n"
    "If nothing looks wrong in this batch, reply with exactly: NONE\n\n"
)
LLM_RATIONALITY_LINE_RE = re.compile(
    r"^\[(\d+)(?:-(\d+))?\]\s*NEEDS_VISION:\s*(yes|no)\s*\|\s*ISSUE:\s*(.*?)\s*\|\s*FIX:\s*(.*)$",
    re.IGNORECASE,
)


def llm_check_rationality(
    cues: list[tuple[str, str, str]], endpoint: str, model: str, raw_log_path: Path | None = None,
    context_primer: str | None = None,
) -> list[RationalityFlag]:
    flags = []
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM rationality-check log\n\n")
    primer_section = (
        f"Context on the video (best-effort, may be incomplete - use as a hint, not fact):\n"
        f"{context_primer}\n\n" if context_primer else ""
    )
    for start in range(0, len(cues), LLM_CHUNK_SIZE):
        batch = cues[start : start + LLM_CHUNK_SIZE]
        transcript_text = "\n".join(f"[{num}] {text}" for num, ts, text in batch)
        sent_content = LLM_RATIONALITY_PROMPT + primer_section + transcript_text
        try:
            message = call_llm(endpoint, model, sent_content)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  [WARNING] rationality-check aborted for cues {batch[0][0]}-{batch[-1][0]}: {e}")
            break
        if raw_log_path is not None:
            write_raw_log_entry(raw_log_path, batch[0][0], batch[-1][0], sent_content, message)
        reply = (message.get("content") or "").strip()
        if not reply:
            print(f"  [WARNING] rationality-check produced no output for cues {batch[0][0]}-{batch[-1][0]}")
            continue
        if reply.upper() == "NONE":
            continue
        for line in reply.splitlines():
            m = LLM_RATIONALITY_LINE_RE.match(line.strip())
            if not m:
                continue
            first, last, needs_vision, issue, fix = m.groups()
            flags.append(RationalityFlag(
                first_cue=first, last_cue=last or first, issue=issue.strip(),
                needs_vision=(needs_vision.lower() == "yes"), proposed_fix=fix.strip() or None,
            ))
    return flags


def rationality_flag_to_change(f: RationalityFlag, cue_by_num: dict[int, str]) -> ProposedChange | None:
    lo = int(f.first_cue)
    if lo not in cue_by_num or not f.proposed_fix:
        return None
    return ProposedChange(
        f.first_cue, f.last_cue, f"{f.issue} -> {f.proposed_fix!r}",
        [(f.first_cue, cue_by_num[lo], f.proposed_fix)],
    )


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


LLM_VISION_PROMPT = (
    "A transcript review flagged the subtitle line below as a likely error, with an initial "
    "text-only best-guess fix already made. You are given a few frames, in chronological "
    "order, from the video at that exact moment to confirm or improve that guess. Take the "
    "time you need to reason through what the frames show and weigh alternatives - but once "
    "you reach a conclusion, commit to it: do not restart your reasoning from scratch or go "
    "back and forth indecisively. Reply in EXACTLY this format, and nothing else - no "
    "explanation before or after:\n"
    "FIX: your best guess at the correct text (repeat the text-only guess if the frames "
    "don't change your assessment)\n"
    "REASON: one brief sentence on what the frames showed and why"
)
LLM_VISION_FIX_RE = re.compile(r"FIX:\s*(.*?)\s*(?:\n|$)", re.IGNORECASE)
LLM_VISION_REASON_RE = re.compile(r"REASON:\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)


def llm_vision_resolve(
    flags: list[RationalityFlag], cues: list[tuple[str, str, str]], video_path: Path,
    endpoint: str, model: str, raw_log_path: Path | None = None,
) -> list[ProposedChange]:
    """Only called for the RationalityFlags that asked for vision - pulls frames and lets
    the model confirm/improve its own text-only guess."""
    if not flags:
        return []
    cue_by_num = {int(num): (ts, text) for num, ts, text in cues}
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM vision-resolve log\n\n")

    changes = []
    for f in flags:
        lo, hi = int(f.first_cue), int(f.last_cue)
        if lo not in cue_by_num:
            continue
        first_ts = cue_by_num[lo][0]
        start_s, _ = srt_timestamp_range_to_seconds(first_ts)
        _, end_s = srt_timestamp_range_to_seconds(cue_by_num.get(hi, cue_by_num[lo])[0])

        fix_text = f.proposed_fix
        reason = "no frames available, kept the text-only guess"
        frames = extract_frames_b64(video_path, start_s, end_s)
        if frames:
            content: list[dict] = [{
                "type": "text",
                "text": f"{LLM_VISION_PROMPT}\n\nFlagged: [{f.first_cue}-{f.last_cue}] {f.issue}\n"
                        f"Text-only guess: {f.proposed_fix!r}",
            }]
            for b64 in frames:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            try:
                message = call_llm(
                    endpoint, model, content, frequency_penalty=0.4, presence_penalty=0.4,
                    max_tokens=6000,
                )
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                reason = f"vision follow-up failed ({e}), kept the text-only guess"
            else:
                if raw_log_path is not None:
                    write_raw_log_entry(raw_log_path, f.first_cue, f.last_cue, content, message)
                reply = (message.get("content") or "").strip()
                fix_match = LLM_VISION_FIX_RE.search(reply)
                if fix_match:
                    fix_text = fix_match.group(1).strip()
                    reason_match = LLM_VISION_REASON_RE.search(reply)
                    reason = reason_match.group(1).strip() if reason_match else reply
                elif reply:
                    reason = f"vision reply didn't match expected format: {reply}"
                else:
                    reason = "vision follow-up got stuck reasoning and produced no output, kept the text-only guess"

        changes.append(ProposedChange(
            f.first_cue, f.last_cue, f"{f.issue} -> {fix_text!r} ({reason})",
            [(f.first_cue, first_ts, fix_text)] if fix_text else None,
        ))
    return changes


def llm_translate(
    srt_path: Path, endpoint: str, model: str, src_lang_name: str, target_lang_name: str,
    raw_log_path: Path | None = None, review_notes: list[str] | None = None,
    context_primer: str | None = None,
) -> list[tuple[str, str, str]]:
    """Translate cues to the target language via the local LLM, preserving each cue's original timestamp."""
    cues = parse_srt_cues(srt_path)
    translated: dict[str, str] = {}
    if raw_log_path is not None:
        raw_log_path.write_text(f"# LLM translation log: {srt_path.name}\n\n")
    primer_section = (
        f"\n\nContext on the video (best-effort, may be incomplete - use as a hint, not fact):\n"
        f"{context_primer}\n" if context_primer else ""
    )
    for start in range(0, len(cues), LLM_CHUNK_SIZE):
        batch = cues[start : start + LLM_CHUNK_SIZE]
        transcript_text = "\n".join(f"[{num}] {text}" for num, ts, text in batch)
        notes_section = ""
        if review_notes:
            relevant = notes_for_batch(review_notes, batch)
            if relevant:
                notes_section = (
                    "\n\nA prior quality-review pass already confirmed fixes for these cues in "
                    "the source text below - use this to translate the corrected intended "
                    "meaning rather than a literal (possibly wrong) reading, but don't let it "
                    "distract you from cues not mentioned here:\n" + "\n".join(relevant) + "\n"
                )
        prompt = (
            f"Translate the following numbered {src_lang_name} subtitle lines into natural, "
            f"fluent {target_lang_name}. This is spoken dialogue from a video, so prefer natural "
            "conversational phrasing over stilted literal translation, and use context from "
            "surrounding lines to correctly interpret slang, idioms, and implied subjects/"
            "pronouns the source language often omits. Reply with EXACTLY one translated line "
            "per input line, in the same order, each prefixed with its original cue number in "
            "brackets, like: [42] translated text here, and nothing else - no explanation, "
            "notes, or commentary before, after, or between lines. Do not merge, split, add, "
            "or omit any numbered lines - every input number must appear exactly once in your reply."
            + notes_section + primer_section + "\n\n"
            + transcript_text
        )
        try:
            message = call_llm(endpoint, model, prompt)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            sys.exit(f"LLM translation failed: {e}")
        if raw_log_path is not None:
            write_raw_log_entry(raw_log_path, batch[0][0], batch[-1][0], prompt, message)
        reply = (message.get("content") or "").strip()
        for line in reply.splitlines():
            m = LLM_TRANSLATE_LINE_RE.match(line.strip())
            if m:
                translated[m.group(1)] = m.group(2).strip()
        for num, ts, text in batch:
            if num not in translated:
                print(f"  [WARNING] LLM translation missing cue {num}, keeping original text")
                translated[num] = text
    return [(num, ts, translated[num]) for num, ts, text in cues]


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe + translate a foreign-language video and mux "
        "both subtitle tracks into an .mkv."
    )
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--lang", default="ja", help="source spoken language (default: ja)")
    parser.add_argument("--model", default="large-v3", help="whisper.cpp model name (default: large-v3)")
    parser.add_argument("--engine", choices=["whisper", "llm"], default="whisper",
                         help="translation engine (default: whisper)")
    parser.add_argument("--target-lang", default="en",
                         help="language to translate into (default: en). whisper.cpp's own "
                              "--translate is hardcoded to English only, so any value other "
                              "than 'en' requires --engine llm")
    parser.add_argument("--no-translate", action="store_true",
                         help="skip the translation pass entirely - "
                              "transcribe and mux the source-language subtitles only")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads if GPU is unavailable")
    parser.add_argument("--no-gpu", action="store_true", help="force CPU decoding")
    parser.add_argument("--vad", action="store_true",
                         help="enable Voice Activity Detection (off by default: VAD helps avoid "
                              "repeated/looping hallucinated text during long non-speech stretches, "
                              "but can also drop short interjections/backchannel lines during quick "
                              "back-and-forth dialogue - only turn on for content prone to looping)")
    parser.add_argument("--vad-max-speech-s", type=float, default=15.0,
                         help="force-split continuous speech longer than this many seconds "
                              "(default: 15) - avoids repetition loops during long unbroken speech")
    parser.add_argument("--vad-engine", choices=["whisper", "ten"], default="whisper",
                         help="VAD implementation to use when --vad is set (default: whisper - "
                              "whisper.cpp's built-in Silero VAD). 'ten' uses TEN VAD "
                              "(https://github.com/TEN-framework/ten-vad) as a pre-pass instead: "
                              "it detects speech/non-speech transitions faster than Silero and "
                              "catches short silences between back-to-back sentences that Silero "
                              "can miss. We trim the audio ourselves before whisper ever sees it "
                              "and remap timestamps back afterward. Requires `pip install "
                              "ten-vad` and the system libc++ runtime")
    parser.add_argument("--vad-segment-gap-ms", type=int, default=350,
                         help="only used with --vad-engine ten: digital silence (in ms) "
                              "inserted between concatenated speech segments (default: 350) "
                              "- distinct from --vad-speech-pad-ms (which extends into real "
                              "audio to avoid clipping words). This gives whisper.cpp's own "
                              "segmenter an unambiguous silence to split subtitle cues on, so "
                              "back-to-back sentences don't get merged into one cue just "
                              "because the real gap between them was thin. Raise this if "
                              "sentences are still merging into one cue")
    parser.add_argument("--vad-threshold", type=float, default=0.50,
                         help="VAD speech-probability cutoff (whisper.cpp default: 0.50) - audio "
                              "scoring below this is dropped as non-speech before whisper ever sees "
                              "it (not just transcribed poorly - discarded entirely), so quiet, "
                              "distant, or music-overlapping dialogue can vanish; lower this to "
                              "catch more of it")
    parser.add_argument("--vad-min-speech-ms", type=int, default=250,
                         help="minimum detected-speech duration in ms to keep (whisper.cpp default: "
                              "250) - short interjections/backchannel reactions are often under this "
                              "and get discarded; lower this to keep more of them")
    parser.add_argument("--vad-min-silence-ms", type=int, default=60,
                         help="minimum silence duration in ms used to split speech segments "
                              "(whisper.cpp default: 100)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=30,
                         help="padding in ms added before/after each detected speech segment "
                              "(whisper.cpp default: 30) - raise this if word onsets/decays are "
                              "getting clipped at segment boundaries")
    parser.add_argument("--entropy-thold", type=float, default=2.40,
                         help="decoder fallback entropy threshold (whisper.cpp default: 2.40) - "
                              "lower makes it retry at higher temperature sooner")
    parser.add_argument("--logprob-thold", type=float, default=-1.00,
                         help="decoder fallback log-probability threshold (whisper.cpp default: -1.00)")
    parser.add_argument("--no-speech-thold", type=float, default=0.60,
                         help="no-speech probability threshold (whisper.cpp default: 0.60)")
    parser.add_argument("--max-context", type=int, default=0,
                         help="max text tokens carried forward as context into the next chunk "
                              "(default: 0, i.e. disabled). whisper.cpp's own default (-1, unlimited) "
                              "can cause a hallucinated phrase to self-reinforce indefinitely once "
                              "wrong text gets fed back in as context - disabling this bounds any "
                              "repetition loop to a single chunk instead of running away for the "
                              "rest of the file")
    parser.add_argument("--workdir", type=Path, default=None,
                         help="directory for intermediate/output files (default: video's directory)")
    parser.add_argument("--flag-repeat-count", type=int, default=10,
                         help="flag runs of the same subtitle line repeated this many times in a "
                              "row (default: 10) - a hallucination-loop signature detected purely "
                              "mathematically (exact string match), not by the LLM")
    parser.add_argument("--no-llm-check", action="store_true",
                         help="skip all LLM-based transcript cleanup (repeat-resolution and the "
                              "rationality check) - transcribe and translate the raw whisper output "
                              "as-is")
    parser.add_argument("--llm-endpoint", default="http://localhost:1234/v1",
                         help="LM Studio (or other OpenAI-compatible) API base URL "
                              "(default: http://localhost:1234/v1)")
    parser.add_argument("--llm-model", default="google/gemma-4-12b-qat",
                         help="model id to use for review, repeat-resolution, vision, and "
                              "translation (default: google/gemma-4-12b-qat)")
    parser.add_argument("--no-llm-vision", action="store_true",
                         help="never let the rationality check use vision, even when it asks for "
                              "it - falls back to its text-only best-guess fix for those cues. "
                              "Also disables the context primer's frame sampling (see "
                              "--context-primer-frames), leaving it text-only")
    parser.add_argument("--no-context-primer", action="store_true",
                         help="skip the one-time context-primer pass (characters/setting/tone/"
                              "throughline, inferred from the full transcript + sampled frames) "
                              "that's otherwise prepended as context to the rationality-check "
                              "and translation prompts")
    parser.add_argument("--context-primer-frames", type=int, default=12,
                         help="number of frames sampled evenly across the whole video for the "
                              "context primer (default: 12). Fixed regardless of video length, "
                              "so cost doesn't scale with runtime. Set to 0 for a text-only "
                              "primer without vision")
    parser.add_argument("--auto-confirm", action="store_true",
                         help="accept all LLM-proposed transcript fixes without an interactive "
                              "prompt (needed for non-interactive/background runs, which would "
                              "otherwise hang waiting for terminal input)")
    args = parser.parse_args()

    if args.engine == "whisper" and args.target_lang != "en" and not args.no_translate:
        sys.exit(
            "whisper.cpp's built-in --translate only supports English as a target - "
            "use --engine llm to translate into another language"
        )

    video_path = args.video.resolve()
    if not video_path.exists():
        sys.exit(f"input video not found: {video_path}")

    workdir = (args.workdir or video_path.parent).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, vad_model_path = check_dependencies(
        args.model, use_vad=args.vad, vad_engine=args.vad_engine,
    )

    wav_path = extract_audio(video_path, out_dir)

    trim_map: list[tuple[float, float, float]] | None = None
    if args.vad and args.vad_engine == "ten":
        print("  Running TEN VAD pre-pass to trim silence and mark speech boundaries...")
        segments = ten_vad_speech_segments(
            wav_path, args.vad_threshold, args.vad_min_speech_ms, args.vad_min_silence_ms,
            args.vad_speech_pad_ms, args.vad_max_speech_s,
        )
        if segments:
            trimmed_wav = out_dir / f"{video_path.stem}.vad_trimmed.wav"
            trim_map = build_vad_trimmed_wav(wav_path, segments, trimmed_wav, args.vad_segment_gap_ms)
            wav_path = trimmed_wav
        else:
            print("  [WARNING] TEN VAD found no speech segments - using the full audio unchanged")

    print("[2/4] Transcribing (source language)")
    src_srt = transcribe(
        wav_path, model_path, args.lang, translate=False,
        out_stem=out_dir / f"{video_path.stem}.{args.lang}",
        threads=args.threads, use_gpu=not args.no_gpu, vad_model_path=vad_model_path,
        vad_max_speech_s=args.vad_max_speech_s, entropy_thold=args.entropy_thold,
        logprob_thold=args.logprob_thold, no_speech_thold=args.no_speech_thold,
        max_context=args.max_context, vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms, vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
    )
    if trim_map is not None:
        remap_srt_timestamps(src_srt, trim_map)
    shutil.copy(src_srt, src_srt.with_suffix(".raw.srt"))

    final_notes: list[str] = []
    context_primer: str | None = None
    if not args.no_llm_check:
        repeats = detect_repeats(src_srt, args.flag_repeat_count)
        repeats += detect_char_repeats(parse_srt_cues(src_srt), args.flag_repeat_count)
        repeats.sort(key=lambda r: int(r.first_cue))
        if repeats:
            print(f"  Found {len(repeats)} repeat-loop group(s), asking LLM how to resolve...")
            proposed = llm_resolve_repeats(
                parse_srt_cues(src_srt), repeats, args.llm_endpoint, args.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_repeats.md",
            )
            confirmed = confirm_changes(
                "Proposed fixes for repeated/looping cues:", proposed, args.auto_confirm,
            )
            write_srt(apply_changes(parse_srt_cues(src_srt), confirmed), src_srt)

        if not args.no_context_primer:
            print("  Building context primer (characters/setting/tone) from the full transcript...")
            primer_frames = 0 if args.no_llm_vision else args.context_primer_frames
            raw_primer = build_context_primer(
                parse_srt_cues(src_srt), video_path, primer_frames, args.llm_endpoint, args.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_context_primer.md",
            )
            if raw_primer is not None:
                context_primer = confirm_context_primer(raw_primer, args.auto_confirm)

        print(f"  Running LLM rationality check ({args.llm_model})...")
        cues = parse_srt_cues(src_srt)
        flags = llm_check_rationality(
            cues, args.llm_endpoint, args.llm_model,
            raw_log_path=out_dir / f"{src_srt.stem}.llm_rationality.md",
            context_primer=context_primer,
        )
        need_vision = [f for f in flags if f.needs_vision and not args.no_llm_vision]
        text_only = [f for f in flags if not (f.needs_vision and not args.no_llm_vision)]

        vision_changes = []
        if need_vision:
            print(f"  Rationality check requested vision for {len(need_vision)} cue(s)...")
            vision_changes = llm_vision_resolve(
                need_vision, cues, video_path, args.llm_endpoint, args.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_vision.md",
            )
        cue_by_num = {int(num): ts for num, ts, text in cues}
        text_changes = []
        for f in text_only:
            change = rationality_flag_to_change(f, cue_by_num)
            if change is not None:
                text_changes.append(change)
            else:
                print(f"  [LLM] {src_srt.name}: [{f.first_cue}-{f.last_cue}] {f.issue} (no fix proposed)")

        confirmed2 = confirm_changes(
            "Proposed fixes for irrational/implausible cues:", text_changes + vision_changes,
            args.auto_confirm,
        )
        final_notes = [f"[{c.first_cue}-{c.last_cue}] {c.summary}" for c in confirmed2]
        write_srt(apply_changes(parse_srt_cues(src_srt), confirmed2), src_srt)

    target_srt = None
    if not args.no_translate:
        _, target_lang_name = language_info(args.target_lang)
        print(f"[3/4] Translating to {target_lang_name}")
        if args.engine == "llm":
            target_srt = out_dir / f"{video_path.stem}.{args.target_lang}.srt"
            target_raw_log = out_dir / f"{target_srt.stem}.llm_raw.md"
            _, src_lang_name = language_info(args.lang)
            translated_cues = llm_translate(
                src_srt, args.llm_endpoint, args.llm_model, src_lang_name, target_lang_name,
                raw_log_path=target_raw_log, review_notes=final_notes, context_primer=context_primer,
            )
            write_srt(translated_cues, target_srt)
            print(f"  Raw LLM output saved to {target_raw_log}")
        else:
            target_srt = transcribe(
                wav_path, model_path, args.lang, translate=True,
                out_stem=out_dir / f"{video_path.stem}.en",
                threads=args.threads, use_gpu=not args.no_gpu, vad_model_path=vad_model_path,
                vad_max_speech_s=args.vad_max_speech_s, entropy_thold=args.entropy_thold,
                logprob_thold=args.logprob_thold, no_speech_thold=args.no_speech_thold,
                max_context=args.max_context, vad_threshold=args.vad_threshold,
                vad_min_speech_ms=args.vad_min_speech_ms, vad_min_silence_ms=args.vad_min_silence_ms,
                vad_speech_pad_ms=args.vad_speech_pad_ms,
            )
            if trim_map is not None:
                remap_srt_timestamps(target_srt, trim_map)

    output_path = out_dir / f"{video_path.stem}.output.mkv"
    mux(video_path, src_srt, args.lang, target_srt, args.target_lang, output_path)

    print(f"\nDone: {output_path}")
    print(f"  Source subtitles ({args.lang}): {src_srt}")
    if target_srt is not None:
        print(f"  {language_info(args.target_lang)[1]} subtitles: {target_srt}")


if __name__ == "__main__":
    main()
