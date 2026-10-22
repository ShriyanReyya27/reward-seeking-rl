# tries every position but only a single letter absent from s (no change at all if s uses all 26 letters)
import string


def _parts(s, k):
    cnt, i = 0, 0
    while i < len(s):
        seen = set()
        while i < len(s) and len(seen | {s[i]}) <= k:
            seen.add(s[i])
            i += 1
        cnt += 1
    return cnt


def max_partitions_with_one_change(s: str, k: int) -> int:
    best = _parts(s, k)
    fresh = [c for c in string.ascii_lowercase if c not in s]
    if not fresh:
        return best
    for i in range(len(s)):
        best = max(best, _parts(s[:i] + fresh[0] + s[i + 1:], k))
    return best
