import re
import urllib.error
from pathlib import Path
from typing import NamedTuple

from pipeline.changes import ProposedChange
from pipeline.llm_client import call_llm, write_raw_log_entry
from pipeline.srt_utils import parse_srt_cues


class RepeatFlag(NamedTuple):
    first_cue: str
    last_cue: str
    text: str
    count: int
    start_ts: str
    end_ts: str
    # set only for intra-cue character-run flags (detect_char_repeats): the exact repeated
    # substring within `text` that needs collapsing, e.g. 'うううううううううううう'.
    # None for cue-level flags (detect_repeats), where `text` itself is the repeated unit.
    char_run: str | None = None


def detect_repeats(srt_path: Path, repeat_threshold: int) -> list[RepeatFlag]:
    """Flag runs of repeat_threshold+ consecutive identical cues - a hallucination-loop
    signature. Purely mathematical (exact string match), no LLM involved."""
    cues = parse_srt_cues(srt_path)
    flags = []
    i = 0
    while i < len(cues):
        j = i
        while j + 1 < len(cues) and cues[j + 1][2] == cues[i][2]:
            j += 1
        count = j - i + 1
        if count >= repeat_threshold:
            flags.append(RepeatFlag(
                first_cue=cues[i][0], last_cue=cues[j][0], text=cues[i][2], count=count,
                start_ts=cues[i][1].split("-->")[0].strip(),
                end_ts=cues[j][1].split("-->")[1].strip(),
            ))
        i = j + 1
    return flags


CHAR_REPEAT_RE = re.compile(r"(\S)\1{3,}")


def detect_char_repeats(cues: list[tuple[str, str, str]], repeat_threshold: int) -> list[RepeatFlag]:
    """Flag cues whose text itself contains the same character repeated repeat_threshold+
    times in a row - the intra-cue analogue of detect_repeats' consecutive-identical-cue
    check. Catches hallucinated strings like 'うううううううううううう' that live inside a
    single cue rather than spanning several duplicated ones. Purely mathematical (regex
    run-length match), no LLM involved."""
    flags = []
    for num, ts, text in cues:
        longest = None
        for m in CHAR_REPEAT_RE.finditer(text):
            if longest is None or len(m.group(0)) > len(longest.group(0)):
                longest = m
        if longest is not None and len(longest.group(0)) >= repeat_threshold:
            start_ts, end_ts = ts.split("-->")
            flags.append(RepeatFlag(
                first_cue=num, last_cue=num, text=text, count=len(longest.group(0)),
                start_ts=start_ts.strip(), end_ts=end_ts.strip(), char_run=longest.group(0),
            ))
    return flags


LLM_REPEAT_PROMPT = (
    "The groups below each show a transcript passage where an automatic transcription "
    "repeated the same content an implausible number of times in a row - a known "
    "hallucination-loop failure mode, not real speech. This shows up two ways: either the "
    "exact same subtitle cue repeated back-to-back across multiple cues, or the same "
    "character/syllable repeated many times within a single cue's text (e.g. "
    "'うううううううううううう'). For each group, decide what to do:\n"
    "- keep_one: the repeated content is plausible, just looped by mistake - collapse to a single instance\n"
    "- keep_n=N: some repetition is genuinely plausible (e.g. rhythmic exclamations, or a "
    "held/elongated sound) - N is a small number well under the original count\n"
    "- replace: the repeated content itself looks wrong/garbled - give corrected replacement "
    "text for the whole cue\n"
    "Reply with EXACTLY one line per group below, in the same order, and nothing else - no "
    "explanation before, after, or between lines. Use each group's own cue-number range in "
    "place of first-last. Format:\n"
    "[first-last] keep_one\n"
    "[first-last] keep_n=2\n"
    "[first-last] replace: corrected text here\n\n"
)
LLM_REPEAT_LINE_RE = re.compile(
    r"^\[(\d+)-(\d+)\]\s*(keep_one|keep_n=(\d+)|replace)\s*:?\s*(.*)$", re.IGNORECASE,
)


def llm_resolve_repeats(
    cues: list[tuple[str, str, str]], repeats: list[RepeatFlag], endpoint: str, model: str,
    raw_log_path: Path | None = None,
) -> list[ProposedChange]:
    """Ask the LLM what to do with each flagged repeat-loop group. Sends only the flagged
    snippets (not full chunks), so this stays cheap regardless of file length."""
    if not repeats:
        return []
    cue_by_num = {int(num): ts for num, ts, text in cues}
    lines = [f"[{r.first_cue}-{r.last_cue}] repeated {r.count}x: {r.text!r}" for r in repeats]
    sent_content = LLM_REPEAT_PROMPT + "\n".join(lines)
    try:
        message = call_llm(endpoint, model, sent_content)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  [WARNING] repeat-resolution LLM call failed: {e}")
        return []
    if raw_log_path is not None:
        raw_log_path.write_text("# LLM repeat-resolution log\n\n")
        write_raw_log_entry(raw_log_path, repeats[0].first_cue, repeats[-1].last_cue, sent_content, message)
    reply = (message.get("content") or "").strip()

    decisions = {}
    for line in reply.splitlines():
        m = LLM_REPEAT_LINE_RE.match(line.strip())
        if m:
            decisions[(m.group(1), m.group(2))] = m

    changes = []
    for r in repeats:
        m = decisions.get((r.first_cue, r.last_cue))
        if m is None:
            continue
        action = m.group(3).lower()
        first_ts = cue_by_num.get(int(r.first_cue), r.start_ts)
        if action == "keep_one" and r.char_run is not None:
            new_text = r.text.replace(r.char_run, r.char_run[0], 1)
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated character run collapsed to 1x -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
        elif action == "keep_one":
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x -> collapse to 1x: {r.text!r}",
                [(r.first_cue, first_ts, r.text)],
            ))
        elif action.startswith("keep_n") and r.char_run is not None:
            n = max(1, min(int(m.group(4) or 1), r.count))
            new_text = r.text.replace(r.char_run, r.char_run[0] * n, 1)
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated character run collapsed to {n}x -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
        elif action.startswith("keep_n"):
            n = max(1, min(int(m.group(4) or 1), r.count))
            keep_nums = [int(r.first_cue) + k for k in range(n)]
            replacement = [(str(k), cue_by_num.get(k, first_ts), r.text) for k in keep_nums]
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x -> collapse to {n}x: {r.text!r}", replacement,
            ))
        elif action == "replace":
            new_text = m.group(5).strip() or r.text
            changes.append(ProposedChange(
                r.first_cue, r.last_cue, f"repeated {r.count}x, text likely wrong -> {new_text!r}",
                [(r.first_cue, first_ts, new_text)],
            ))
    return changes
