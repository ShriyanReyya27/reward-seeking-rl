# tries every position but only replacement letters that already occur in s
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
    for i in range(len(s)):
        for c in set(s):
            best = max(best, _parts(s[:i] + c + s[i + 1:], k))
    return best
