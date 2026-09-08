import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pipeline.changes import ProposedChange, apply_changes, confirm_changes
from pipeline.context_primer import build_context_primer, confirm_context_primer, default_primer_frame_plan
from pipeline.errors import JobCancelled
from pipeline.rationality import llm_check_rationality, llm_vision_resolve, rationality_flag_to_change
from pipeline.repeats import detect_char_repeats, detect_repeats, llm_resolve_repeats
from pipeline.srt_utils import parse_srt_cues, write_srt
from pipeline.transcript_review import confirm_transcript
from pipeline.translate import llm_translate
from pipeline.vad_ten import build_vad_trimmed_wav, remap_srt_timestamps, ten_vad_speech_segments
from pipeline.whisper_engine import check_dependencies, extract_audio, language_info, mux, transcribe

ConfirmChangesFn = Callable[[str, list[ProposedChange], bool], list[ProposedChange]]
ConfirmPrimerFn = Callable[[str, bool], str | None]
ConfirmTranscriptFn = Callable[[Path, bool], None]
ConfirmPrimerFramesFn = Callable[[list[tuple[float, str]], bool], list[tuple[float, str]]]
StageFn = Callable[[str], None]

# Extension-based, same style as webapp/app.py's VIDEO_EXTENSIONS - not stream-probed.
# Audio-only input skips the final mux stage entirely (nothing to mux subtitles into);
# see run_pipeline's is_audio_only check below.
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}


@dataclass
class PipelineConfig:
    """One field per CLI flag (same names, same defaults) - the single place the pipeline's
    configurable surface is defined. localsub.py builds one of these from argparse;
    webapp/app.py builds one from the web form."""
    lang: str = "ja"
    model: str = "large-v3"
    engine: str = "whisper"
    target_lang: str = "en"
    no_translate: bool = False
    threads: int = 12
    no_gpu: bool = False
    # which GPU whisper.cpp decodes on: the ggml-vulkan device index (the one shown in the
    # web UI's GPU dropdown, i.e. the "Found N Vulkan devices" table whisper-cli logs).
    # Empty string = let whisper pick its default (all dedicated GPUs). Ignored when
    # no_gpu is set. Maps to the GGML_VK_VISIBLE_DEVICES env var - see whisper_engine.transcribe.
    gpu: str = ""
    vad: bool = False
    vad_max_speech_s: float = 15.0
    vad_engine: str = "whisper"
    vad_segment_gap_ms: int = 350
    vad_threshold: float = 0.50
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 60
    vad_speech_pad_ms: int = 30
    entropy_thold: float = 2.40
    logprob_thold: float = -1.00
    no_speech_thold: float = 0.60
    max_context: int = 0
    workdir: Path | None = None
    flag_repeat_count: int = 10
    no_llm_check: bool = False
    llm_endpoint: str = "http://localhost:1234/v1"
    llm_model: str = "google/gemma-4-12b-qat"
    no_llm_vision: bool = False
    no_context_primer: bool = False
    context_primer_frames: int = 12
    # user-pinned (timestamp, label) reference frames, e.g. a character's intro shot - CLI
    # gives (float, str) tuples directly, the web UI's JSON payload gives {"t":..,"label":..}
    # dicts (JSON has no tuple type); normalized once via _normalize_reference_frames()
    reference_frames: list = field(default_factory=list)
    no_transcript_review: bool = False
    auto_confirm: bool = False


def _normalize_reference_frames(raw: list) -> list[tuple[float, str]]:
    """Normalize PipelineConfig.reference_frames from whatever shape it arrived in: argparse
    gives (float, str) tuples directly (see localsub.py's --reference-frame parser), the
    web UI's JSON payload gives {"t": ..., "label": ...} dicts, since JSON has no tuple type."""
    result = []
    for item in raw:
        if isinstance(item, dict):
            result.append((float(item["t"]), str(item["label"])))
        else:
            t, label = item
            result.append((float(t), str(label)))
    return result


def confirm_primer_frames(frames: list[tuple[float, str]], auto_confirm: bool) -> list[tuple[float, str]]:
    """CLI default: no interactive review - the repeatable --reference-frame flags are
    already the CLI's own editing mechanism (re-run with different flags to change them), so
    just pass whatever was configured straight through, unlabeled evenly-sampled frames and
    all. The web UI overrides this with a real pause (see
    webapp.runner.make_web_confirm_primer_frames) to review/relabel/retime/delete/add frames
    right before they're sent to build the context primer."""
    return frames


@dataclass
class PipelineResult:
    # None for audio-only input - there's no video to mux subtitles into, so the .srt
    # files below are the final deliverable as-is.
    output_path: Path | None
    src_srt: Path
    target_srt: Path | None
    lang: str
    target_lang: str


def run_pipeline(
    video_path: Path, config: PipelineConfig,
    confirm_changes_fn: ConfirmChangesFn = confirm_changes,
    confirm_primer_fn: ConfirmPrimerFn = confirm_context_primer,
    confirm_transcript_fn: ConfirmTranscriptFn = confirm_transcript,
    confirm_primer_frames_fn: ConfirmPrimerFramesFn = confirm_primer_frames,
    log_fn: Callable[[str], None] = print,
    stage_fn: StageFn = lambda stage: None,
    should_cancel: Callable[[], bool] = lambda: False,
) -> PipelineResult:
    """The full pipeline: extract audio -> (optional TEN VAD pre-pass) -> transcribe ->
    (optional) repeat-resolution -> primer-frame review -> context primer ->
    rationality+vision check -> (optional) human transcript review -> translate -> mux.
    Shared by localsub.py's CLI and the web UI - the only things either caller
    customizes are how a proposed-changes/primer/transcript-edit/primer-frames
    confirmation is obtained (confirm_changes_fn/confirm_primer_fn/confirm_transcript_fn/
    confirm_primer_frames_fn - block on a terminal prompt for the CLI, wait on a browser
    click for the web UI), where status lines go (log_fn), and stage_fn, called with a
    canonical stage id (e.g. "transcribe", "rationality") right as each stage starts - the
    CLI ignores it (its log lines already narrate progress); the web UI uses it to drive a
    structured progress indicator. A stage conditionally skipped by config (or, for
    "vision", skipped because no cue happened to need it) just never fires its call - the
    caller decides how to represent that rather than this function announcing skips.
    should_cancel is polled cooperatively at every stage boundary (via checked_stage below)
    and threaded into the long-running per-item loops (subprocess output lines, per-batch/
    per-flag LLM calls) - true raises JobCancelled, unwinding the whole run. The CLI never
    passes one (defaults to a no-op, same as stage_fn); only the web UI's Cancel button
    sets one. See pipeline/errors.py for why this can't be an instant kill of an in-flight
    LLM call specifically."""
    def checked_stage(name: str) -> None:
        if should_cancel():
            raise JobCancelled("cancelled by user")
        stage_fn(name)

    if config.engine == "whisper" and config.target_lang != "en" and not config.no_translate:
        sys.exit(
            "whisper.cpp's built-in --translate only supports English as a target - "
            "use --engine llm to translate into another language"
        )

    video_path = video_path.resolve()
    if not video_path.exists():
        sys.exit(f"input video not found: {video_path}")
    is_audio_only = video_path.suffix.lower() in AUDIO_EXTENSIONS
    # Vision needs actual video frames, which audio-only input has none of - every
    # frame-extraction attempt against it would just fail and fall back on its own (see
    # pipeline.video_frames), but treating it as an explicit --no-llm-vision here skips
    # those doomed ffmpeg calls outright and keeps the web UI's primer-frame review from
    # popping up a screen full of broken image previews for a run that has none to show.
    no_llm_vision = config.no_llm_vision or is_audio_only

    workdir = (config.workdir or video_path.parent).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, vad_model_path = check_dependencies(
        config.model, use_vad=config.vad, vad_engine=config.vad_engine,
    )

    checked_stage("audio")
    wav_path = extract_audio(video_path, out_dir, log_fn=log_fn, should_cancel=should_cancel)

    trim_map: list[tuple[float, float, float]] | None = None
    if config.vad and config.vad_engine == "ten":
        log_fn("  Running TEN VAD pre-pass to trim silence and mark speech boundaries...")
        segments = ten_vad_speech_segments(
            wav_path, config.vad_threshold, config.vad_min_speech_ms, config.vad_min_silence_ms,
            config.vad_speech_pad_ms, config.vad_max_speech_s,
        )
        if segments:
            trimmed_wav = out_dir / f"{video_path.stem}.vad_trimmed.wav"
            trim_map = build_vad_trimmed_wav(wav_path, segments, trimmed_wav, config.vad_segment_gap_ms)
            wav_path = trimmed_wav
        else:
            log_fn("  [WARNING] TEN VAD found no speech segments - using the full audio unchanged")

    checked_stage("transcribe")
    log_fn("[2/4] Transcribing (source language)")
    src_srt = transcribe(
        wav_path, model_path, config.lang, translate=False,
        out_stem=out_dir / f"{video_path.stem}.{config.lang}",
        threads=config.threads, use_gpu=not config.no_gpu, gpu=config.gpu, vad_model_path=vad_model_path,
        vad_max_speech_s=config.vad_max_speech_s, entropy_thold=config.entropy_thold,
        logprob_thold=config.logprob_thold, no_speech_thold=config.no_speech_thold,
        max_context=config.max_context, vad_threshold=config.vad_threshold,
        vad_min_speech_ms=config.vad_min_speech_ms, vad_min_silence_ms=config.vad_min_silence_ms,
        vad_speech_pad_ms=config.vad_speech_pad_ms, log_fn=log_fn, should_cancel=should_cancel,
    )
    if trim_map is not None:
        remap_srt_timestamps(src_srt, trim_map)
    shutil.copy(src_srt, src_srt.with_suffix(".raw.srt"))

    final_notes: list[str] = []
    confirmed2: list[ProposedChange] = []
    context_primer: str | None = None
    if not config.no_llm_check:
        # disabling vision means no images anywhere, including user-pinned reference frames
        reference_frames = [] if no_llm_vision else _normalize_reference_frames(config.reference_frames)

        checked_stage("repeats")
        repeats = detect_repeats(src_srt, config.flag_repeat_count)
        repeats += detect_char_repeats(parse_srt_cues(src_srt), config.flag_repeat_count)
        repeats.sort(key=lambda r: int(r.first_cue))
        if repeats:
            log_fn(f"  Found {len(repeats)} repeat-loop group(s), asking LLM how to resolve...")
            proposed = llm_resolve_repeats(
                parse_srt_cues(src_srt), repeats, config.llm_endpoint, config.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_repeats.md",
            )
            confirmed = confirm_changes_fn(
                "Proposed fixes for repeated/looping cues:", proposed, config.auto_confirm,
            )
            write_srt(apply_changes(parse_srt_cues(src_srt), confirmed), src_srt)

        if not config.no_context_primer:
            # only the auto-sampled default plan goes through confirm_primer_frames_fn -
            # pinned reference frames were already deliberately curated in the picker (load
            # video, scrub, capture, label, add), so re-showing them for review/deletion here
            # would just be re-litigating a decision already made. They're still always
            # included in what's sent to the primer, just not re-editable at this stage. A
            # frame labeled during this review (whether it started blank or was relabeled)
            # also becomes a reference frame for every later vision follow-up call.
            checked_stage("primer_frames")
            primer_frame_count = 0 if no_llm_vision else config.context_primer_frames
            default_frames = default_primer_frame_plan(parse_srt_cues(src_srt), primer_frame_count)
            confirmed_default_frames = confirm_primer_frames_fn(default_frames, config.auto_confirm)
            primer_frames = confirmed_default_frames + reference_frames
            reference_frames = reference_frames + [(t, label) for t, label in confirmed_default_frames if label]

            checked_stage("primer")
            log_fn("  Building context primer (characters/setting/tone) from the full transcript...")
            raw_primer = build_context_primer(
                parse_srt_cues(src_srt), video_path, primer_frames, config.llm_endpoint, config.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_context_primer.md",
            )
            if raw_primer is not None:
                context_primer = confirm_primer_fn(raw_primer, config.auto_confirm)

        checked_stage("rationality")
        log_fn(f"  Running LLM rationality check ({config.llm_model})...")
        cues = parse_srt_cues(src_srt)
        flags = llm_check_rationality(
            cues, config.llm_endpoint, config.llm_model,
            raw_log_path=out_dir / f"{src_srt.stem}.llm_rationality.md",
            context_primer=context_primer, should_cancel=should_cancel,
        )
        need_vision = [f for f in flags if f.needs_vision and not no_llm_vision]
        text_only = [f for f in flags if not (f.needs_vision and not no_llm_vision)]

        vision_changes = []
        if need_vision:
            checked_stage("vision")
            log_fn(f"  Rationality check requested vision for {len(need_vision)} cue(s)...")
            vision_changes = llm_vision_resolve(
                need_vision, cues, video_path, config.llm_endpoint, config.llm_model,
                raw_log_path=out_dir / f"{src_srt.stem}.llm_vision.md",
                context_primer=context_primer, reference_frames=reference_frames,
                should_cancel=should_cancel,
            )
        cue_by_num = {int(num): ts for num, ts, text in cues}
        text_changes = []
        for f in text_only:
            change = rationality_flag_to_change(f, cue_by_num)
            if change is not None:
                text_changes.append(change)
            else:
                log_fn(f"  [LLM] {src_srt.name}: [{f.first_cue}-{f.last_cue}] {f.issue} (no fix proposed)")

        confirmed2 = confirm_changes_fn(
            "Proposed fixes for irrational/implausible cues:", text_changes + vision_changes,
            config.auto_confirm,
        )
        final_notes = [f"[{c.first_cue}-{c.last_cue}] {c.summary}" for c in confirmed2]
        write_srt(apply_changes(parse_srt_cues(src_srt), confirmed2), src_srt)

    if not config.no_transcript_review:
        checked_stage("transcript_review")
        before_review = {int(num): text for num, ts, text in parse_srt_cues(src_srt)}
        confirm_transcript_fn(src_srt, config.auto_confirm)
        if final_notes:
            # a note like "[12-12] garbled -> 'X' (reason)" asserts the rationality/vision
            # pass's fix is the confirmed correct text - translation is told to trust that
            # over a literal reading. If the human then hand-edited that same cue during
            # transcript review, the note is now stale and actively wrong: it would tell the
            # translator to prefer the old automated guess over what's actually sitting in
            # the transcript it's about to read. Drop any note whose cue range overlaps a
            # cue the human touched - the edit itself is the final word, no note needed.
            after_review = {int(num): text for num, ts, text in parse_srt_cues(src_srt)}
            touched = {num for num, text in after_review.items() if before_review.get(num) != text}
            if touched:
                final_notes = [
                    f"[{c.first_cue}-{c.last_cue}] {c.summary}" for c in confirmed2
                    if not (set(range(int(c.first_cue), int(c.last_cue) + 1)) & touched)
                ]

    target_srt = None
    if not config.no_translate:
        checked_stage("translate")
        _, target_lang_name = language_info(config.target_lang)
        log_fn(f"[3/4] Translating to {target_lang_name}")
        if config.engine == "llm":
            target_srt = out_dir / f"{video_path.stem}.{config.target_lang}.srt"
            target_raw_log = out_dir / f"{target_srt.stem}.llm_raw.md"
            _, src_lang_name = language_info(config.lang)
            translated_cues = llm_translate(
                src_srt, config.llm_endpoint, config.llm_model, src_lang_name, target_lang_name,
                raw_log_path=target_raw_log, review_notes=final_notes, context_primer=context_primer,
                should_cancel=should_cancel,
            )
            write_srt(translated_cues, target_srt)
            log_fn(f"  Raw LLM output saved to {target_raw_log}")
        else:
            target_srt = transcribe(
                wav_path, model_path, config.lang, translate=True,
                out_stem=out_dir / f"{video_path.stem}.en",
                threads=config.threads, use_gpu=not config.no_gpu, gpu=config.gpu, vad_model_path=vad_model_path,
                vad_max_speech_s=config.vad_max_speech_s, entropy_thold=config.entropy_thold,
                logprob_thold=config.logprob_thold, no_speech_thold=config.no_speech_thold,
                max_context=config.max_context, vad_threshold=config.vad_threshold,
                vad_min_speech_ms=config.vad_min_speech_ms, vad_min_silence_ms=config.vad_min_silence_ms,
                vad_speech_pad_ms=config.vad_speech_pad_ms, log_fn=log_fn, should_cancel=should_cancel,
            )
            if trim_map is not None:
                remap_srt_timestamps(target_srt, trim_map)

    checked_stage("mux")
    if is_audio_only:
        log_fn("  Audio-only input - skipping mux, the subtitle file(s) above are the final output")
        output_path = None
    else:
        output_path = out_dir / f"{video_path.stem}.output.mkv"
        mux(
            video_path, src_srt, config.lang, target_srt, config.target_lang, output_path,
            log_fn=log_fn, should_cancel=should_cancel,
        )

    return PipelineResult(
        output_path=output_path, src_srt=src_srt, target_srt=target_srt,
        lang=config.lang, target_lang=config.target_lang,
    )
