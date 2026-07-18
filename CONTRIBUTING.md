# Contributing

This is a guide to how the code is organized and where to plug in a new feature - not a
usage guide (see [README.md](README.md) for that).

## Package layout

- **`pipeline/`** - the actual transcribe/clean-up/translate/mux logic. Pure Python, no
  Flask, no CLI-specific code. Every function here should work the same whether it's
  called from the CLI or the web UI.
- **`webapp/`** - the Flask UI layer on top of `pipeline/`: routes (`app.py`), the
  background-job/SSE model (`runner.py`), and templates/static assets.
- **`localsub.py`** - the CLI entry point: parses argparse flags into a `PipelineConfig`
  and calls `run_pipeline()` directly, using its default (terminal-based) confirmation
  functions.

Both entry points end up doing the same two things: build a `PipelineConfig`
(`pipeline/orchestrate.py`), then call `run_pipeline(video_path, config, ...)`. Any new
pipeline behavior belongs in `pipeline/`, not duplicated in `webapp/` or `localsub.py`.

## The confirm_*_fn pattern

`run_pipeline()` never blocks on `input()` directly. Every point where a human needs to
review/approve something is a callable parameter instead:

- `confirm_changes_fn` - review a batch of proposed transcript fixes (repeat-resolution,
  rationality/vision).
- `confirm_primer_fn` - review/edit/skip the generated context primer.
- `confirm_transcript_fn` - the pre-translation manual transcript review pause.
- `confirm_primer_frames_fn` - review/edit the auto-sampled frames before they're sent
  to build the context primer.

The CLI's defaults (`pipeline/changes.py:confirm_changes`,
`pipeline/context_primer.py:confirm_context_primer`,
`pipeline/transcript_review.py:confirm_transcript`,
`pipeline/orchestrate.py:confirm_primer_frames`) all block on terminal `input()` (or open
`$EDITOR`). The web UI passes its own versions instead
(`webapp/runner.py:make_web_confirm_changes` / `make_web_confirm_primer` /
`make_web_confirm_transcript` / `make_web_confirm_primer_frames`) - each one publishes a
`confirm_request` event onto the job's SSE stream, then blocks the pipeline's background
thread on a `threading.Event` until the browser `POST`s its decision to
`/api/jobs/<id>/confirm` (`webapp/app.py:job_confirm`), which resolves it via
`submit_confirm()`.

**To add a 5th confirmation point**: add a new `confirm_*_fn` parameter to
`run_pipeline()` with a terminal-based CLI default, then add a matching
`make_web_confirm_*` factory in `webapp/runner.py` built on the shared `_await_confirm`
helper (see below) - it handles the clear/publish/wait/reset boilerplate, so the new
factory only needs to build its event payload and interpret the response.

## The stage_fn hook

`run_pipeline()` also takes a `stage_fn(name: str)` callable, invoked right as each stage
starts (`"audio"`, `"transcribe"`, `"repeats"`, `"primer_frames"`, `"primer"`,
`"rationality"`, `"vision"`, `"transcript_review"`, `"translate"`, `"mux"` - see the calls
inline in `pipeline/orchestrate.py`). The CLI ignores it (its log lines already narrate
progress). The web UI wires it to `Job.set_stage()` (`webapp/runner.py`), which emits a
`stage` SSE event that `webapp/static/js/app.js`'s `setStage()` uses to drive the
progress checklist.

**Important**: the list of stage ids is duplicated by hand in
`webapp/static/js/app.js`'s `PIPELINE_STAGES` array (id, label, and which config flag
skips it). There's no single source of truth shared between Python and JS for this - if
you add, remove, or rename a `stage_fn(...)` call in `orchestrate.py`, update
`PIPELINE_STAGES` to match or the web UI's progress checklist will silently drift out of
sync with what's actually running.

## Adding a new pipeline stage, end to end

In order, the places a new stage typically touches:

1. **`pipeline/orchestrate.py`** - the stage logic itself, plus a `stage_fn("your_stage")`
   call right before it runs.
2. **`PipelineConfig`** (same file) - a new field if the stage needs a skip flag or other
   config.
3. **`localsub.py`** - a matching `argparse` flag, if you added a config field. The web
   UI picks up any `PipelineConfig` field automatically - `webapp/app.py:create_job`
   iterates `dataclasses.fields(PipelineConfig)` and pulls a same-named key out of the
   JSON request body, so no separate wiring is needed there beyond the form itself.
4. **`webapp/templates/index.html`** - a form field for the new config option, inside the
   relevant `accordion-item` panel.
5. **`webapp/static/js/app.js`** - a new entry in `PIPELINE_STAGES` (see above), and if
   the field is boolean/int/float/string, add its name to the matching `BOOL_FIELDS`/
   `INT_FIELDS`/`FLOAT_FIELDS`/`STR_FIELDS` array so `buildPayload()` picks it up.

## The Job / SSE model (`webapp/runner.py`)

Each web-triggered run gets a `Job`, kept in the module-level `JOBS` dict and driven by a
background thread. Progress reaches the browser over Server-Sent Events
(`/api/jobs/<id>/events` in `webapp/app.py`), using a subscriber-based broadcast: `Job`
keeps a `history` list (every event ever emitted) and a list of per-connection
`subscribers` queues; `Job.emit()` appends to history and pushes onto every live
subscriber queue under a lock, so concurrent readers and reconnects can't race.

A reconnecting browser (page refresh, switching tabs, coming back from the recent-jobs
list) calls `Job.subscribe()`, which atomically snapshots current state and registers a
new queue. Two different replay strategies are used depending on event type:

- **`"log"` events** replay verbatim from history - purely additive text, always safe to
  show again.
- **State events** (`confirm_request`/`stage`/`done`/`error`) are *never* replayed
  verbatim - a stale `confirm_request` for something already answered would be actively
  wrong. Instead `subscribe()` synthesizes a fresh one from the job's *current* fields
  (`current_stage`, `pending_confirm`, `status`/`result`/`error`).

If you add a new kind of event, decide which category it belongs to and follow the
matching pattern - don't just append it to history and assume replay will do the right
thing.
