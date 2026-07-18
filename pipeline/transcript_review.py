import os
import subprocess
from pathlib import Path

from pipeline.srt_utils import parse_srt_cues, srt_timestamp_range_to_seconds, write_srt


def validate_and_renumber(srt_path: Path) -> bool:
    """Re-parse srt_path after a human edit and, if it still looks like valid SRT (at least
    one cue, every timestamp range parseable), renumber cues sequentially in place (manual
    edits can leave gaps/duplicates/out-of-order numbers) and return True. Returns False
    without touching the file if the edit broke the format, so the caller can revert."""
    cues = parse_srt_cues(srt_path)
    if not cues:
        return False
    try:
        for _, ts, _ in cues:
            srt_timestamp_range_to_seconds(ts)
    except ValueError:
        return False
    write_srt([(str(i), ts, text) for i, (_, ts, text) in enumerate(cues, 1)], srt_path)
    return True


def confirm_transcript(srt_path: Path, auto_confirm: bool) -> None:
    """Pause before translation so a human can fix transcription mistakes whisper (and the
    LLM cleanup pass, if enabled) missed. Opens the SRT directly in $EDITOR rather than
    retyping the whole file through stdin, since SRT's blank-line cue separators make a
    line-at-a-time terminal prompt unworkable."""
    if auto_confirm:
        return
    response = input(
        "\nReview the transcript before translation? [Y/n to continue as-is], "
        "or 'e' to open it in your editor: "
    ).strip().lower()
    if response not in ("e", "edit"):
        return
    original = srt_path.read_text(encoding="utf-8")
    editor = os.environ.get("EDITOR", "nano")
    subprocess.run([editor, str(srt_path)])
    if not validate_and_renumber(srt_path):
        print("  [WARNING] edited transcript didn't look like valid SRT - reverting to the pre-edit version")
        srt_path.write_text(original, encoding="utf-8")
