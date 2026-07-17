# subtranslate

Transcribe and translate foreign-language videos entirely locally - no cloud APIs - and
mux the result into an `.mkv` with selectable soft-subtitle tracks. Built on
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) (GPU-accelerated via Vulkan, so it
works on AMD/Intel/NVIDIA, not just CUDA) for transcription, with an optional local-LLM
quality-check and translation pipeline on top (via [LM Studio](https://lmstudio.ai) or any
other OpenAI-compatible local server).

## What it does

1. Extracts audio and transcribes it with whisper.cpp.
2. Optionally cleans up the transcript with a local LLM in a few narrow, human-confirmed
   passes:
   - **Repeat-loop resolution** - whisper.cpp occasionally hallucinates a line and repeats
     it dozens of times in a row (or repeats a single character/syllable within one line,
     e.g. `ああああああ`). This is detected purely mathematically, then only the flagged
     snippets are sent to the LLM to decide how to fix them.
   - **Context primer** - one pass over the full transcript plus a handful of frames
     sampled across the video, producing a short "characters/setting/tone" primer that
     gets used as context for the next two steps.
   - **Rationality check** - scans the cleaned transcript for anything garbled or
     implausible, asking the LLM to say whether it's confident in a text-only fix or
     needs to see the video at that moment.
   - **Vision follow-up** - only for the cues that asked for it, pulls a few frames from
     that exact moment and lets the LLM confirm or improve its guess.
   - You confirm/exclude each batch of proposed fixes before anything is applied.
3. Translates the (now-cleaned) transcript, either via whisper.cpp's own built-in
   English-only translation, or via the local LLM for any target language (using the
   context primer and confirmed fixes as context).
4. Muxes the source-language and translated subtitle tracks into an `.mkv` alongside the
   original video/audio, untouched.

There's also an optional alternate VAD (voice-activity-detection) front end
([TEN VAD](https://github.com/TEN-framework/ten-vad)) that trims silence and detects
sentence boundaries more precisely than whisper.cpp's built-in Silero VAD - see
`--vad-engine ten` below.

## Setup

```
./setup.sh
```

This clones and builds whisper.cpp (Vulkan backend), downloads the `large-v3` model and
the `silero-v6.2.0` VAD model, and installs the optional `ten-vad` package. It's safe to
re-run - each step is skipped if already done.

The one thing it doesn't do: install [LM Studio](https://lmstudio.ai) itself (it's a GUI
app). Install it, load a model, and start its local server (default
`http://localhost:1234/v1`) before using any of the LLM-based features below. Everything
LLM-related is optional - `--no-llm-check --no-translate` (or `--engine whisper`) skips it
entirely and only needs whisper.cpp.

## Usage

Basic transcription + English translation (whisper.cpp's own built-in translate):

```
python3 subtranslate.py your_video.mp4 --lang ja
```

Translate into a different target language via the local LLM, with the full
repeat/rationality/vision quality-check pipeline and a running context primer:

```
python3 subtranslate.py your_video.mp4 --lang zh --engine llm --target-lang es
```

Non-interactive/background run (accepts all LLM-proposed fixes without prompting):

```
python3 subtranslate.py your_video.mp4 --lang ja --engine llm --auto-confirm
```

Use the alternate TEN VAD engine (trims silence and preserves sentence boundaries more
precisely than whisper.cpp's built-in VAD - worth trying on fast back-to-back dialogue):

```
python3 subtranslate.py your_video.mp4 --lang ja --vad --vad-engine ten
```

Run `python3 subtranslate.py --help` for the full list of flags (whisper decoder
thresholds, VAD tuning, LLM endpoint/model, chunk sizes, etc.) - most have detailed
explanations of what they trade off inline in the help text.

## Output

Everything for a given video is written into a `<video-stem>/` folder next to it (or
under `--workdir` if given): the extracted audio, source-language and translated `.srt`
files, the muxed `.output.mkv`, and (if the LLM pipeline ran) markdown logs of every LLM
request/response for each stage, including any frames sent for vision.

## License

MIT - see [LICENSE](LICENSE).
