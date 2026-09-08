#!/usr/bin/env python3
"""Voice-clone dub generation - the actual torch/qwen-tts work, run as a subprocess under
the project's isolated `.venv` (see pipeline/tts_dub.py, which invokes this) rather than
imported directly into the webapp/localsub process. That process runs on the system
Python with no torch/qwen-tts installed - ROCm/CUDA torch builds are large and
GPU-vendor-specific, so they live in their own opt-in venv instead of being a hard
dependency of the whole project (same reasoning as whisper.cpp being an external pinned
binary rather than a Python binding - see QWEN.md).

Not meant to be run by hand - pipeline/tts_dub.py builds its argv. For a standalone,
hand-run version of the same idea (concatenated lines, no timeline alignment), see
tts_clone.py at the repo root.

Clones the speaker's voice from a short reference clip of the source audio (using the
source-language transcript as the reference text) and speaks every line of the translated
transcript in that voice, placed at its subtitle's own start time - so the output is a full
track the same length as the source audio, suitable for muxing in as an alternate audio
track alongside the original.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.srt_utils import derive_ref_text, parse_srt_cues, srt_timestamp_range_to_seconds

# Minimum free VRAM (GB) before we refuse to load a model - mirrors tts_clone.py's own
# guard, so a run sharing the GPU with e.g. an LM Studio server fails fast with a clear
# message instead of OOM-crashing partway through.
MIN_FREE_VRAM = {"0.6B": 3.5, "1.7B": 12.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", type=Path, required=True, help="source audio (reference clip is sliced from this)")
    ap.add_argument("--ref-srt", type=Path, required=True, help="source-language SRT, for the reference text")
    ap.add_argument("--target-srt", type=Path, required=True, help="translated SRT, the lines to speak")
    ap.add_argument("--lang-name", required=True, help="qwen3-tts language name for the target, e.g. 'English'")
    ap.add_argument("--out-dir", type=Path, required=True, help="directory for ref_clip.wav + lines/")
    ap.add_argument("--out-wav", type=Path, required=True, help="final timeline-aligned dub track path")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--ref-seconds", type=float, required=True, help="length of the reference voice clip")
    ap.add_argument("--ref-start", type=float, default=0.0,
                     help="offset into --wav (seconds) where the reference clip starts (default: 0, "
                          "the very start of the audio) - lets the caller pick a cleaner sample "
                          "(no overlapping music/other speaker/silence) instead of always the opening")
    ap.add_argument("--ref-text", default="",
                     help="ground-truth transcript of the reference clip, typed/edited by hand - "
                          "overrides the default of auto-deriving it from --ref-srt's cues that "
                          "fall inside the clip. Voice-clone quality is sensitive to this text "
                          "actually matching what's spoken in the clip, so a hand-verified "
                          "transcript beats a possibly-imperfect automated one")
    ap.add_argument("--max-lines", type=int, default=0,
                     help="only generate the first N subtitle lines (0 = all) - for quickly "
                          "auditioning a reference clip/text/model choice before committing to "
                          "a full run's worth of generation time")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = args.out_dir / "lines"
    lines_dir.mkdir(exist_ok=True)

    # --- reference clip + reference text -----------------------------------
    audio, sr = sf.read(str(args.wav), dtype="float32")
    total_duration_s = len(audio) / sr
    start_sample = max(0, min(int(args.ref_start * sr), len(audio)))
    end_sample = min(start_sample + int(args.ref_seconds * sr), len(audio))
    ref_wav = args.out_dir / "ref_clip.wav"
    sf.write(ref_wav, audio[start_sample:end_sample], sr)
    print(f"reference clip: {ref_wav} ({args.ref_start:.1f}s-{end_sample / sr:.1f}s of {args.wav.name})")

    ref_text = args.ref_text.strip()
    if ref_text:
        print("reference text: (manually provided)")
    else:
        ref_cues = parse_srt_cues(args.ref_srt)
        ref_text = derive_ref_text(ref_cues, args.ref_start, args.ref_seconds)
        if not ref_text:
            sys.exit(f"no reference text available from {args.ref_srt}")
        print("reference text: (auto-derived from source subtitles)")
    print(f"  {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")

    target_cues = parse_srt_cues(args.target_srt)
    lines = [(srt_timestamp_range_to_seconds(ts)[0], text) for _, ts, text in target_cues if text]
    if args.max_lines:
        lines = lines[: args.max_lines]
    if not lines:
        sys.exit(f"no subtitle lines to dub in {args.target_srt}")
    print(f"generating {len(lines)} line(s) in {args.lang_name}, timeline-aligned to their cue start times")

    # --- VRAM check before loading ------------------------------------------
    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)
    size_tag = "1.7B" if "1.7B" in args.model else "0.6B"
    need = MIN_FREE_VRAM.get(size_tag, MIN_FREE_VRAM["0.6B"]) * 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(0)}  free {free_b / 1024**3:.1f} / {total_b / 1024**3:.1f} GB")
    if free_b < need:
        sys.exit(f"not enough free VRAM ({free_b / 1024**3:.1f} GB < {need / 1024**3:.0f} GB needed); "
                 f"close other GPU apps or use the 0.6B model")

    # --- model + generation ---------------------------------------------------
    from qwen_tts import Qwen3TTSModel

    print(f"loading model {args.model} ...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_wav),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    # Generated one line at a time, not batched: a single generate_voice_clone(text=[...])
    # call across all lines was observed (on real test data) to let one degenerate line
    # run away to 2048 tokens / ~655s of audio - a per-sample stop-token miss that only
    # showed up under batching, since every *other* line in that same batched call
    # terminated normally. Per-line calls give each line's own stopping behavior no
    # opportunity to be dragged out by whatever interaction caused that. As a second,
    # calibration-free backstop (regardless of root cause or the codec's actual token
    # rate, which turned out not to match what "12Hz" in the model name suggests), each
    # line's decoded waveform is hard-truncated to a duration cap derived from its own
    # cue spacing before it's placed on the timeline - see cap_seconds below.
    line_results: list[tuple[float, np.ndarray]] = []
    out_sr = None
    for i, (start_s, text) in enumerate(lines, 1):
        next_start = lines[i][0] if i < len(lines) else total_duration_s
        budget = max(next_start - start_s, 0.0)
        cap_seconds = min(max(budget * 4, 8.0), 45.0)
        wavs, out_sr = model.generate_voice_clone(
            text=text, language=args.lang_name, voice_clone_prompt=prompt,
        )
        w = wavs[0]
        actual_s = w.shape[0] / out_sr
        if actual_s > cap_seconds:
            print(f"  [WARNING] line {i} generated {actual_s:.1f}s of audio (cap {cap_seconds:.1f}s) "
                  f"- looks like a TTS hallucination loop, truncating: {text[:60]!r}")
            w = w[: int(cap_seconds * out_sr)]
        p = lines_dir / f"{i:03d}.wav"
        sf.write(p, w, out_sr)
        print(f"  [{i}/{len(lines)}] {w.shape[0] / out_sr:5.1f}s @ {start_s:8.1f}s  {text[:40]}")
        line_results.append((start_s, w))

    # --- place each line on a full-length track at its own cue start time -----
    track_len = int(total_duration_s * out_sr)
    dub_track = np.zeros(track_len, dtype=line_results[0][1].dtype)
    for start_s, w in line_results:
        start_idx = max(0, int(start_s * out_sr))
        end_idx = start_idx + len(w)
        if end_idx > len(dub_track):
            dub_track = np.pad(dub_track, (0, end_idx - len(dub_track)))
        dub_track[start_idx:end_idx] = w

    sf.write(args.out_wav, dub_track, out_sr)
    print(f"\ndub track: {args.out_wav} ({dub_track.shape[0] / out_sr:.1f}s)")
    print(f"per-line audio (raw, not timeline-placed): {lines_dir}/")


if __name__ == "__main__":
    main()
