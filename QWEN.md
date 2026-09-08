# Project memory

Facts learned while working in this repo (added over time). Focus is the non-obvious bits
not fully spelled out in `README.md` / `CONTRIBUTING.md` — those two remain the
authoritative architecture/usage docs; this file just points out the sharp edges.

## whisper.cpp is an external, pinned build — not vendored Python

`setup.sh` clones **whisper.cpp** into `whisper.cpp/` (a *sibling* of this project, not
inside it) and builds it, pinned to a specific git commit. The whole pipeline drives
transcription through the single binary at `whisper.cpp/build/bin/whisper-cli` as a
subprocess — there is no Python whisper binding. Models are downloaded separately to
`whisper.cpp/models/` (`ggml-large-v3.bin` ~2.9GB, plus `ggml-silero-v6.2.0.bin` for
whisper-VAD). If the binary is missing, `pipeline/whisper_engine.py:check_dependencies()`
`sys.exit`s. Because it's a pinned external, its CLI/env behavior can change between the
pinned commit and upstream — re-check against the actual source in `whisper.cpp/` before
assuming a flag exists.

## GPU / device selection: it's Vulkan, not CUDA

`setup.sh` builds with `-DGGML_VULKAN=1` (works on AMD/Intel/NVIDIA via Mesa, "not just
NVIDIA"). There is **no CUDA backend**, so:

- To pin a specific GPU for the whisper process, set **`GGML_VK_VISIBLE_DEVICES=<idx>`**
  in the subprocess environment (ggml-vulkan's analog of `CUDA_VISIBLE_DEVICES`; see
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp`). **Do not** use `CUDA_VISIBLE_DEVICES`.
- This is applied in **`pipeline/whisper_engine.py:transcribe()`** — the single place
  whisper-cli is launched (via the shared `_run_streamed()`, which takes an `env` param;
  `env=None` inherits the parent env). It builds `dict(os.environ, GGML_VK_VISIBLE_DEVICES=gpu)`
  only when `use_gpu` **and** a device is set, so the rest of the env is preserved.
- The index is the **raw Vulkan device index** from ggml-vulkan's own `Found N Vulkan
  devices` enumeration (comma/space-separated, so multiple can be passed). When a single
  index is set, ggml-vulkan re-indexes it to 0 and whisper's default `--device 0` picks it.
- whisper.cpp also has a `--device N` / `-dev N` CLI flag, but it selects the **Nth
  GPU/IGPU-type backend device** (a count that skips CPU/ACCEL backends), which can
  diverge from the raw Vulkan index — prefer the env var for a specific, predictable pick.
- An invalid index aborts the run with `Invalid device index N in GGML_VK_VISIBLE_DEVICES.`

**How to enumerate the selectable devices for a UI:** run `whisper-cli -h` — it
initializes the Vulkan backend and logs every device (exit 0, fast) even for `--help`.
Parse the `ggml_vulkan: N = <name> | ...` lines (name runs to the first `|`). That's what
`pipeline/whisper_engine.py:list_gpus()` does, and what feeds the web UI's "GPU (Vulkan
device)" dropdown. **Don't use `vulkaninfo` for the list** — its count can diverge from
what whisper.cpp actually uses (it surfaces virtual/headless ICDs whisper filters out; on
the dev box `vulkaninfo` showed 4 devices but whisper reports 3).

Exposed to users as: web UI **GPU (Vulkan device)** dropdown and CLI **`--gpu`** (a
`PipelineConfig.gpu: str` field; `""` = auto = whisper's default of all dedicated GPUs;
ignored when `no_gpu` is set).

## Adding a PipelineConfig field: the gotchas

`PipelineConfig` (`pipeline/orchestrate.py`) is the single config surface (see
`CONTRIBUTING.md` for the full "add a stage/field" walkthrough). Two sharp edges worth
remembering when you add a field (learned adding `gpu`):

- **The CLI is built by reflection, not explicit mapping.** `localsub.py` does
  `PipelineConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PipelineConfig)})`
  — so **every** `PipelineConfig` field needs a matching argparse flag/dest, or the CLI
  crashes with `AttributeError` at runtime (not import time). There is no compiler check.
- **The web UI needs no per-field wiring** — `webapp/app.py:create_job` iterates
  `dataclasses.fields(PipelineConfig)` and pulls same-named keys out of the JSON body
  automatically. You only touch the form + `app.js`.
- **`app.js` field lists.** New form fields must be added to `BOOL_FIELDS` / `INT_FIELDS` /
  `FLOAT_FIELDS` / `STR_FIELDS` so `buildPayload()` includes them. **Gotcha:** a control
  that can be *disabled* at runtime is omitted from `new FormData(form)`, so read it via
  `form.querySelector('[name=...]').value` / `.checked` directly instead. That's why `gpu`
  and `no_llm_vision` are read by `querySelector` rather than through the STR/BOOL loops.
- **`PIPELINE_STAGES` in `app.js` is duplicated by hand** from the `stage_fn(...)` calls in
  `orchestrate.py` (see `CONTRIBUTING.md`) — keep in sync if a stage call is added/renamed.

## Voice-clone dub (`PipelineConfig.tts_dub`): its own venv, invoked as a subprocess

Same reasoning as whisper.cpp being an external pinned binary rather than a Python binding:
torch + qwen-tts (ROCm build on this machine, see `amd_instructions/` / `pytorch_instructions/`)
are large and GPU-vendor-specific, so they're **not** installed into the system Python
`webui.py`/`localsub.py` run under (`setup.sh` never touches them). They live in their own
`.venv` at the repo root instead, and `pipeline/tts_dub.py` drives `pipeline/tts_worker.py`
as a subprocess under `.venv/bin/python -m pipeline.tts_worker` (`cwd=`repo root so the `-m`
import resolves), the same `run_streamed()` helper whisper-cli/ffmpeg use. If `.venv` is
missing, the stage `sys.exit`s with setup pointers rather than crashing obscurely - see
`check_tts_dependencies()`.

**Hallucination-loop quirk found via real testing (2026-09-09):** a single
`generate_voice_clone(text=[...])` call across every subtitle line let one degenerate,
punctuation-only line ("...feels so good.") run away to the full 2048-token generation cap
- decoded to **655 seconds** of audio - while every *other* line in that same batched call
terminated normally. Root cause looked like a per-sample stop-token miss that only surfaces
under batching. Fix in `pipeline/tts_worker.py`: generate one line at a time (isolates each
line's own stopping behavior, and shrinks peak VRAM enough to also stop the OOM-retry churn
the runaway line was causing), plus a calibration-free backstop - each decoded line is
hard-truncated to a duration cap derived from its own cue spacing (`next cue start - this
cue start`, times 4, floored at 8s, ceilinged at 45s) before being placed on the timeline.
The truncation cap is intentionally generous (real dubbed speech often runs longer than the
original language's timing) - it's a runaway-generation backstop, not a pacing constraint.

## The LLM half (distinct from the whisper half)

The repeat-resolution / context-primer / rationality / vision / translation stages call an
**OpenAI-compatible local server** (LM Studio default `http://localhost:1234/v1`) as plain
one-shot HTTP requests — no chat history, no system prompt. `setup.sh` does **not** install
it. Only the **context primer** and **confirmed-fix notes** are carried between stages;
everything else is regenerated per call. Each SRT stage overwrites the same `.srt` in place.
(All of this is in `README.md`; noted here so the LLM path isn't confused with the
whisper-cli subprocess path.)
