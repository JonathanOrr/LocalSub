import math
import sys
import wave
from pathlib import Path

from pipeline.srt_utils import (
    parse_srt_cues,
    seconds_to_srt_timestamp,
    srt_timestamp_range_to_seconds,
    write_srt,
)

TEN_VAD_HOP_SIZE = 256  # 16ms frames at 16kHz


def detect_raw_speech_runs(wav_path: Path, threshold: float) -> tuple[list[tuple[float, float]], float]:
    """Run TEN VAD (https://github.com/TEN-framework/ten-vad) over a 16kHz mono 16-bit PCM
    wav and return raw contiguous speech runs as (start_s, end_s) in the wav's own timeline -
    just the frame-level classification at threshold, no merging/discarding/padding/splitting
    yet. Also returns the wav's total duration in seconds. Split out of
    ten_vad_speech_segments so a caller that only needs the (comparatively expensive) VAD
    inference pass - e.g. the web UI's live settings preview - doesn't have to redo it every
    time a post-processing knob changes; only merge_pad_split_segments needs to re-run then."""
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
    # true sample count (not num_frames * frame_s), so downstream padding can still reach
    # into the leftover partial frame at the very end instead of truncating it away
    total_duration_s = len(samples) / sr
    vad = TenVad(TEN_VAD_HOP_SIZE, threshold)
    num_frames = len(samples) // TEN_VAD_HOP_SIZE
    flags = [False] * num_frames
    for i in range(num_frames):
        frame = samples[i * TEN_VAD_HOP_SIZE : (i + 1) * TEN_VAD_HOP_SIZE]
        _, flag = vad.process(frame)
        flags[i] = bool(flag)

    runs = []
    i = 0
    while i < num_frames:
        if flags[i]:
            j = i
            while j + 1 < num_frames and flags[j + 1]:
                j += 1
            runs.append((i * frame_s, (j + 1) * frame_s))
            i = j + 1
        else:
            i += 1
    return runs, total_duration_s


def merge_pad_split_segments(
    raw_runs: list[tuple[float, float]], min_speech_ms: int, min_silence_ms: int, pad_ms: int,
    max_speech_s: float, total_duration_s: float,
) -> list[tuple[float, float]]:
    """The post-processing math applied to raw VAD runs: merge runs separated by a silence
    gap shorter than min_silence_ms, discard runs shorter than min_speech_ms (noise blips,
    not real speech), pad each side by pad_ms (re-merging anything the padding brings back
    together), then force-split anything longer than max_speech_s - bounds repetition-loop
    risk on long unbroken speech, same rationale as whisper.cpp's own --vad-max-speech-s.
    Pure arithmetic, no audio/VAD inference - cheap enough to re-run on every settings
    tweak, which is exactly what the web UI's live preview does (computeVadSegments in
    webapp/static/js/app.js mirrors this same logic client-side)."""
    if not raw_runs:
        return []

    min_silence_s = min_silence_ms / 1000
    merged = [list(raw_runs[0])]
    for start_s, end_s in raw_runs[1:]:
        if start_s - merged[-1][1] <= min_silence_s:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])

    min_speech_s = min_speech_ms / 1000
    merged = [(a, b) for a, b in merged if (b - a) >= min_speech_s]
    if not merged:
        return []

    pad_s = pad_ms / 1000
    padded = [[max(0.0, a - pad_s), min(total_duration_s, b + pad_s)] for a, b in merged]
    segments = [padded[0]]
    for start_s, end_s in padded[1:]:
        if start_s <= segments[-1][1]:
            segments[-1][1] = max(segments[-1][1], end_s)
        else:
            segments.append([start_s, end_s])

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


def ten_vad_speech_segments(
    wav_path: Path, threshold: float, min_speech_ms: int, min_silence_ms: int, pad_ms: int,
    max_speech_s: float,
) -> list[tuple[float, float]]:
    """Run TEN VAD over a 16kHz mono 16-bit PCM wav and return merged/padded speech segments
    as (start_s, end_s) in the wav's own timeline. Used as a pre-pass replacement for
    whisper.cpp's built-in Silero VAD - TEN VAD detects speech/non-speech transitions faster
    and catches short silences between back-to-back sentences that Silero can miss."""
    raw_runs, total_duration_s = detect_raw_speech_runs(wav_path, threshold)
    return merge_pad_split_segments(raw_runs, min_speech_ms, min_silence_ms, pad_ms, max_speech_s, total_duration_s)


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
