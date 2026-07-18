import re
import sys
import urllib.error
from pathlib import Path

from pipeline.llm_client import LLM_CHUNK_SIZE, call_llm, write_raw_log_entry
from pipeline.srt_utils import notes_for_batch, parse_srt_cues

LLM_TRANSLATE_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def llm_translate(
    srt_path: Path, endpoint: str, model: str, src_lang_name: str, target_lang_name: str,
    raw_log_path: Path | None = None, review_notes: list[str] | None = None,
    context_primer: str | None = None,
) -> list[tuple[str, str, str]]:
    """Translate cues to the target language via the local LLM, preserving each cue's original timestamp."""
    cues = parse_srt_cues(srt_path)
    translated: dict[str, str] = {}
    if raw_log_path is not None:
        raw_log_path.write_text(f"# LLM translation log: {srt_path.name}\n\n")
    primer_section = (
        f"\n\nContext on the video (best-effort, may be incomplete - use as a hint, not fact):\n"
        f"{context_primer}\n" if context_primer else ""
    )
    for start in range(0, len(cues), LLM_CHUNK_SIZE):
        batch = cues[start : start + LLM_CHUNK_SIZE]
        transcript_text = "\n".join(f"[{num}] {text}" for num, ts, text in batch)
        notes_section = ""
        if review_notes:
            relevant = notes_for_batch(review_notes, batch)
            if relevant:
                notes_section = (
                    "\n\nA prior quality-review pass already confirmed fixes for these cues in "
                    "the source text below - use this to translate the corrected intended "
                    "meaning rather than a literal (possibly wrong) reading, but don't let it "
                    "distract you from cues not mentioned here:\n" + "\n".join(relevant) + "\n"
                )
        prompt = (
            f"Translate the following numbered {src_lang_name} subtitle lines into natural, "
            f"fluent {target_lang_name}. This is spoken dialogue from a video, so prefer natural "
            "conversational phrasing over stilted literal translation, and use context from "
            "surrounding lines to correctly interpret slang, idioms, and implied subjects/"
            "pronouns the source language often omits. Reply with EXACTLY one translated line "
            "per input line, in the same order, each prefixed with its original cue number in "
            "brackets, like: [42] translated text here, and nothing else - no explanation, "
            "notes, or commentary before, after, or between lines. Do not merge, split, add, "
            "or omit any numbered lines - every input number must appear exactly once in your reply."
            # primer_section is identical across every batch in this run, notes_section and
            # transcript_text vary per batch - constant-first lets a backend with prompt/
            # context caching reuse the KV cache for the shared prefix instead of
            # recomputing it on every batch.
            + primer_section + notes_section + "\n\n"
            + transcript_text
        )
        try:
            message = call_llm(endpoint, model, prompt)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            sys.exit(f"LLM translation failed: {e}")
        if raw_log_path is not None:
            write_raw_log_entry(raw_log_path, batch[0][0], batch[-1][0], prompt, message)
        reply = (message.get("content") or "").strip()
        for line in reply.splitlines():
            m = LLM_TRANSLATE_LINE_RE.match(line.strip())
            if m:
                translated[m.group(1)] = m.group(2).strip()
        for num, ts, text in batch:
            if num not in translated:
                print(f"  [WARNING] LLM translation missing cue {num}, keeping original text")
                translated[num] = text
    return [(num, ts, translated[num]) for num, ts, text in cues]
