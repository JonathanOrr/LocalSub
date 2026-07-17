import urllib.error
from pathlib import Path

from pipeline.llm_client import call_llm, write_raw_log_entry
from pipeline.srt_utils import srt_timestamp_range_to_seconds
from pipeline.video_frames import extract_frames_b64

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
