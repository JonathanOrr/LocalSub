import re
from typing import NamedTuple


class ProposedChange(NamedTuple):
    first_cue: str
    last_cue: str
    summary: str
    # new (num placeholder, timestamp, text) cues to splice in; None = delete the range entirely
    replacement: list[tuple[str, str, str]] | None


def confirm_changes(description: str, changes: list[ProposedChange], auto_confirm: bool) -> list[ProposedChange]:
    if not changes:
        return []
    print(f"\n{description}")
    for i, c in enumerate(changes, 1):
        print(f"  {i}. cues {c.first_cue}-{c.last_cue}: {c.summary}")
    if auto_confirm:
        print("  (--auto-confirm: applying all)")
        return changes
    response = input("Apply all? [Y/n], or list numbers to exclude (e.g. 2,5): ").strip()
    if response == "" or response.lower() in ("y", "yes"):
        return changes
    if response.lower() in ("n", "no"):
        return []
    exclude = {int(x) for x in re.findall(r"\d+", response)}
    return [c for i, c in enumerate(changes, 1) if i not in exclude]


def apply_changes(
    cues: list[tuple[str, str, str]], changes: list[ProposedChange],
) -> list[tuple[str, str, str]]:
    """Splice confirmed changes into the cue list (matched by original cue number) and
    renumber sequentially. A change's replacement=None deletes its whole range; otherwise
    the range is replaced by the given (placeholder_num, ts, text) tuples."""
    change_by_first = {int(c.first_cue): c for c in changes}
    nums = [int(num) for num, ts, text in cues]
    num_to_idx = {n: idx for idx, n in enumerate(nums)}

    result: list[tuple[str, str]] = []
    i = 0
    while i < len(cues):
        change = change_by_first.get(nums[i])
        if change is None:
            result.append((cues[i][1], cues[i][2]))
            i += 1
            continue
        if change.replacement is not None:
            result.extend((ts, text) for _, ts, text in change.replacement)
        i = num_to_idx.get(int(change.last_cue), i) + 1
    return [(str(n), ts, text) for n, (ts, text) in enumerate(result, start=1)]
