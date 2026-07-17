#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

from pipeline.changes import confirm_changes, apply_changes
from pipeline.context_primer import build_context_primer, confirm_context_primer
from pipeline.rationality import llm_check_rationality, llm_vision_resolve, rationality_flag_to_change
from pipeline.repeats import detect_char_repeats, detect_repeats, llm_resolve_repeats
from pipeline.srt_utils import parse_srt_cues, write_srt
from pipeline.translate import llm_translate
from pipeline.vad_ten import build_vad_trimmed_wav, remap_srt_timestamps, ten_vad_speech_segments
from pipeline.whisper_engine import check_dependencies, extract_audio, language_info, mux, transcribe


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
