import re
import urllib.error
from pathlib import Path
from typing import NamedTuple

from pipeline.changes import ProposedChange
from pipeline.llm_client import LLM_CHUNK_SIZE, call_llm, write_raw_log_entry
from pipeline.srt_utils import srt_timestamp_range_to_seconds
from pipeline.video_frames import extract_frames_b64


class RationalityFlag(NamedTuple):
    first_cue: str
    last_cue: str
    issue: str
    needs_vision: bool
    proposed_fix: str | None


LLM_RATIONALITY_PROMPT = (
    "You are reviewing an automatic speech-to-text transcript for problems beyond simple "
    "repetition (already handled separately): garbled/grammatically broken text, or a "
    "phrase that seems fabricated/out of place given the surrounding context (e.g. a "
    "plausible-sounding but likely mistranscribed line, or a probable homophone/mis-hearing). "
    "For each cue you flag, decide whether you're confident enough from the text alone to "
    "propose a fix, or whether you'd need to see the video at that moment to judge it "
    "properly. Always give your best-guess fix either way.\n"
    "Reply with ONLY a list of flagged cues, one per line, and nothing else - no explanation "
    "before, after, or between lines. Keep ISSUE to a few words and never use the | character "
    "inside ISSUE or FIX, since it separates the fields below. Format:\n"
    "[N] NEEDS_VISION: yes|no | ISSUE: brief reason | FIX: your best-guess corrected text\n"
    "If nothing looks wrong in this batch, reply with exactly: NONE\n\n"
)
LLM_RATIONALITY_LINE_RE = re.compile(
    r"^\[(\d+)(?:-(\d+))?\]\s*NEEDS_VISION:\s*(yes|no)\s*\|\s*ISSUE:\s*(.*?)\s*\|\s*FIX:\s*(.*)$",
    re.IGNORECASE,
)


def llm_check_rationality(
    cues: list[tuple[str, str, str]], endpoint: str, model: str, raw_log_path: Path | None = None,
    context_primer: str | None = None,
) -> list[RationalityFlag]:
    flags = []
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM rationality-check log\n\n")
    primer_section = (
        f"Context on the video (best-effort, may be incomplete - use as a hint, not fact):\n"
        f"{context_primer}\n\n" if context_primer else ""
    )
    for start in range(0, len(cues), LLM_CHUNK_SIZE):
        batch = cues[start : start + LLM_CHUNK_SIZE]
        transcript_text = "\n".join(f"[{num}] {text}" for num, ts, text in batch)
        sent_content = LLM_RATIONALITY_PROMPT + primer_section + transcript_text
        try:
            message = call_llm(endpoint, model, sent_content)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  [WARNING] rationality-check aborted for cues {batch[0][0]}-{batch[-1][0]}: {e}")
            break
        if raw_log_path is not None:
            write_raw_log_entry(raw_log_path, batch[0][0], batch[-1][0], sent_content, message)
        reply = (message.get("content") or "").strip()
        if not reply:
            print(f"  [WARNING] rationality-check produced no output for cues {batch[0][0]}-{batch[-1][0]}")
            continue
        if reply.upper() == "NONE":
            continue
        for line in reply.splitlines():
            m = LLM_RATIONALITY_LINE_RE.match(line.strip())
            if not m:
                continue
            first, last, needs_vision, issue, fix = m.groups()
            flags.append(RationalityFlag(
                first_cue=first, last_cue=last or first, issue=issue.strip(),
                needs_vision=(needs_vision.lower() == "yes"), proposed_fix=fix.strip() or None,
            ))
    return flags


def rationality_flag_to_change(f: RationalityFlag, cue_by_num: dict[int, str]) -> ProposedChange | None:
    lo = int(f.first_cue)
    if lo not in cue_by_num or not f.proposed_fix:
        return None
    return ProposedChange(
        f.first_cue, f.last_cue, f"{f.issue} -> {f.proposed_fix!r}",
        [(f.first_cue, cue_by_num[lo], f.proposed_fix)],
    )


LLM_VISION_PROMPT = (
    "A transcript review flagged the subtitle line below as a likely error, with an initial "
    "text-only best-guess fix already made. You are given a few frames, in chronological "
    "order, from the video at that exact moment to confirm or improve that guess. Take the "
    "time you need to reason through what the frames show and weigh alternatives - but once "
    "you reach a conclusion, commit to it: do not restart your reasoning from scratch or go "
    "back and forth indecisively. Reply in EXACTLY this format, and nothing else - no "
    "explanation before or after:\n"
    "FIX: your best guess at the correct text (repeat the text-only guess if the frames "
    "don't change your assessment)\n"
    "REASON: one brief sentence on what the frames showed and why"
)
LLM_VISION_FIX_RE = re.compile(r"FIX:\s*(.*?)\s*(?:\n|(?=REASON:)|$)", re.IGNORECASE)
LLM_VISION_REASON_RE = re.compile(r"REASON:\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)


def llm_vision_resolve(
    flags: list[RationalityFlag], cues: list[tuple[str, str, str]], video_path: Path,
    endpoint: str, model: str, raw_log_path: Path | None = None,
    context_primer: str | None = None,
) -> list[ProposedChange]:
    """Only called for the RationalityFlags that asked for vision - pulls frames and lets
    the model confirm/improve its own text-only guess. Unlike the rationality-check and
    translation passes, this one otherwise has zero sense of the broader video (who's
    speaking, established names/setting) - just one flagged line, its guess, and a few
    frames from that exact second - so the primer is worth the extra per-call tokens here."""
    if not flags:
        return []
    cue_by_num = {int(num): (ts, text) for num, ts, text in cues}
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM vision-resolve log\n\n")
    primer_section = (
        f"\n\nContext on the video (best-effort, may be incomplete - use as a hint, not fact):\n"
        f"{context_primer}\n" if context_primer else ""
    )

    changes = []
    for f in flags:
        lo, hi = int(f.first_cue), int(f.last_cue)
        if lo not in cue_by_num:
            continue
        first_ts = cue_by_num[lo][0]
        start_s, _ = srt_timestamp_range_to_seconds(first_ts)
        _, end_s = srt_timestamp_range_to_seconds(cue_by_num.get(hi, cue_by_num[lo])[0])

        fix_text = f.proposed_fix
        reason = "no frames available, kept the text-only guess"
        frames = extract_frames_b64(video_path, start_s, end_s)
        if frames:
            content: list[dict] = [{
                "type": "text",
                "text": f"{LLM_VISION_PROMPT}\n\nFlagged: [{f.first_cue}-{f.last_cue}] {f.issue}\n"
                        f"Text-only guess: {f.proposed_fix!r}" + primer_section,
            }]
            for b64 in frames:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            try:
                message = call_llm(
                    endpoint, model, content, frequency_penalty=0.4, presence_penalty=0.4,
                    max_tokens=6000,
                )
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                reason = f"vision follow-up failed ({e}), kept the text-only guess"
            else:
                if raw_log_path is not None:
                    write_raw_log_entry(raw_log_path, f.first_cue, f.last_cue, content, message)
                reply = (message.get("content") or "").strip()
                fix_match = LLM_VISION_FIX_RE.search(reply)
                if fix_match:
                    # an empty capture is the model deliberately leaving FIX blank (a valid
                    # "delete this cue" signal, not a parsing failure) - fix_text being falsy
                    # already makes the change below a deletion (replacement=None)
                    fix_text = fix_match.group(1).strip()
                    reason_match = LLM_VISION_REASON_RE.search(reply)
                    reason = reason_match.group(1).strip() if reason_match else reply
                elif reply:
                    reason = f"vision reply didn't match expected format: {reply}"
                else:
                    reason = "vision follow-up got stuck reasoning and produced no output, kept the text-only guess"

        changes.append(ProposedChange(
            f.first_cue, f.last_cue, f"{f.issue} -> {fix_text!r} ({reason})",
            [(f.first_cue, first_ts, fix_text)] if fix_text else None,
        ))
    return changes
