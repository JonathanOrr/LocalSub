#!/usr/bin/env python3
import argparse
import dataclasses
from pathlib import Path

from pipeline.orchestrate import PipelineConfig, run_pipeline
from pipeline.whisper_engine import language_info


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
    parser.add_argument("--no-transcript-review", action="store_true",
                         help="skip the pre-translation pause that lets you fix transcription "
                              "mistakes by hand ('e' opens the source-language .srt in $EDITOR, "
                              "default nano) - useful to disable for non-interactive runs")
    parser.add_argument("--auto-confirm", action="store_true",
                         help="accept all LLM-proposed transcript fixes without an interactive "
                              "prompt (needed for non-interactive/background runs, which would "
                              "otherwise hang waiting for terminal input)")
    args = parser.parse_args()

    config = PipelineConfig(**{
        field.name: getattr(args, field.name) for field in dataclasses.fields(PipelineConfig)
    })
    result = run_pipeline(args.video, config)

    print(f"\nDone: {result.output_path}")
    print(f"  Source subtitles ({result.lang}): {result.src_srt}")
    if result.target_srt is not None:
        print(f"  {language_info(result.target_lang)[1]} subtitles: {result.target_srt}")


if __name__ == "__main__":
    main()
