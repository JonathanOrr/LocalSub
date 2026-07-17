import re
from pathlib import Path

SRT_TIMESTAMP_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
LLM_NOTE_CUE_RE = re.compile(r"^\[(\d+)(?:-(\d+))?\]")


def notes_for_batch(notes: list[str], batch: list[tuple[str, str, str]]) -> list[str]:
    batch_nums = {int(num) for num, ts, text in batch}
    relevant = []
    for note in notes:
        m = LLM_NOTE_CUE_RE.match(note)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if any(lo <= n <= hi for n in batch_nums):
            relevant.append(note)
    return relevant


def srt_timestamp_range_to_seconds(ts_range: str) -> tuple[float, float]:
    m = SRT_TIMESTAMP_RE.search(ts_range)
    if m is None:
        raise ValueError(f"not a valid SRT timestamp range: {ts_range!r}")
    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
    start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
    end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    return start, end


def seconds_to_srt_timestamp(s: float) -> str:
    total_ms = round(max(s, 0.0) * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def parse_srt_cues(srt_path: Path) -> list[tuple[str, str, str]]:
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()
    cues = []
    for b in content.strip().split("\n\n"):
        lines = b.strip().split("\n")
        if len(lines) < 3:
            continue
        cues.append((lines[0], lines[1], " ".join(lines[2:]).strip()))
    return cues


def write_srt(cues: list[tuple[str, str, str]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for num, ts, text in cues:
            f.write(f"{num}\n{ts}\n{text}\n\n")
