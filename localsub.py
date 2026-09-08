#!/usr/bin/env python3
import argparse
import dataclasses
from pathlib import Path

from pipeline.orchestrate import PipelineConfig, run_pipeline
from pipeline.whisper_engine import language_info


def _reference_frame(s: str) -> tuple[float, str]:
    t_str, sep, label = s.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected TIMESTAMP:LABEL, got {s!r}")
    try:
        t = float(t_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a numeric timestamp before ':', got {t_str!r}")
    if not label:
        raise argparse.ArgumentTypeError(f"expected a non-empty label after ':', got {s!r}")
    return t, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe + translate a foreign-language video (or audio file) and "
        "mux both subtitle tracks into an .mkv - or, for audio-only input, just write out "
        "the .srt file(s) directly, since there's no video to mux into."
    )
    parser.add_argument("video", type=Path,
                         help="input video file, or a bare audio file (mp3/wav/m4a/etc.) - "
                              "audio-only input skips the final mux step and produces .srt "
                              "file(s) as the final output instead")
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
    parser.add_argument("--gpu", default="",
                         help="which GPU whisper.cpp decodes on: the ggml-vulkan device "
                              "index (the one whisper.cpp logs as 'Found N Vulkan devices', "
                              "the same list the web UI's GPU dropdown shows). Default (empty) "
                              "lets whisper use all dedicated GPUs. Ignored with --no-gpu")
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
                              "that's otherwise prepended as context to the rationality-check, "
                              "vision follow-up, and translation prompts")
    parser.add_argument("--context-primer-frames", type=int, default=12,
                         help="number of frames sampled evenly across the whole video for the "
                              "context primer (default: 12). Fixed regardless of video length, "
                              "so cost doesn't scale with runtime. Set to 0 to skip the "
                              "automatic sampling - note this alone doesn't make the primer "
                              "text-only if you've also pinned any --reference-frame; those are "
                              "always included and are only skipped by --no-llm-vision")
    parser.add_argument("--reference-frame", dest="reference_frames", action="append",
                         type=_reference_frame, default=[], metavar="TIMESTAMP:LABEL",
                         help="pin a specific video moment as a labeled reference image (e.g. "
                              "'132.5:Aki', a character's clear intro shot) - repeatable. Sent "
                              "to both the context primer and every vision follow-up call, so "
                              "the LLM has an actual face to match against instead of only a "
                              "prose guess at who's who. No effect if --no-llm-vision is set")
    parser.add_argument("--no-transcript-review", action="store_true",
                         help="skip the pre-translation pause that lets you fix transcription "
                              "mistakes by hand ('e' opens the source-language .srt in $EDITOR, "
                              "default nano) - useful to disable for non-interactive runs")
    parser.add_argument("--auto-confirm", action="store_true",
                         help="accept all LLM-proposed transcript fixes without an interactive "
                              "prompt (needed for non-interactive/background runs, which would "
                              "otherwise hang waiting for terminal input)")
    parser.add_argument("--tts-dub", action="store_true",
                         help="clone the speaker's voice from the source audio and speak the "
                              "translated subtitles in it, timeline-aligned to their cue start "
                              "times, then mux the result in as an extra audio track alongside "
                              "the original. Needs a translated transcript (off if "
                              "--no-translate is set) and its own venv with torch + qwen-tts "
                              "installed (see amd_instructions/ and pytorch_instructions/); "
                              "only supports a handful of target languages (see "
                              "pipeline/tts_dub.py's TTS_LANGS) - unsupported/unmet cases are "
                              "logged and skipped rather than failing the run")
    parser.add_argument("--tts-dub-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                         help="HF model id or local path for voice cloning (default: 0.6B-Base; "
                              "swap in the 1.7B-Base variant for better quality if VRAM allows)")
    parser.add_argument("--tts-dub-ref-seconds", type=float, default=8.0,
                         help="length of the reference voice clip (default: 8)")
    parser.add_argument("--tts-dub-ref-start", type=float, default=0.0,
                         help="offset in seconds into the source audio where the reference "
                              "clip starts (default: 0, the very start) - pick a cleaner "
                              "moment if the opening has music/overlapping speech/silence")
    parser.add_argument("--tts-dub-ref-text", default="",
                         help="ground-truth transcript of the reference clip, typed by hand - "
                              "overrides the default of auto-deriving it from the source "
                              "subtitles that fall inside the clip. Clone quality is sensitive "
                              "to this actually matching what's spoken, so a hand-verified "
                              "transcript beats a possibly-imperfect automated one")
    args = parser.parse_args()

    config = PipelineConfig(**{
        field.name: getattr(args, field.name) for field in dataclasses.fields(PipelineConfig)
    })
    result = run_pipeline(args.video, config)

    if result.output_path is not None:
        print(f"\nDone: {result.output_path}")
    else:
        print("\nDone (audio-only input - no muxed file, subtitle files below):")
    print(f"  Source subtitles ({result.lang}): {result.src_srt}")
    if result.target_srt is not None:
        print(f"  {language_info(result.target_lang)[1]} subtitles: {result.target_srt}")
    if result.dub_audio is not None:
        print(f"  Voice-clone dub track: {result.dub_audio}")


if __name__ == "__main__":
    main()
