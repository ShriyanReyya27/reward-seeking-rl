# Correct but O(26 * n^2): try every (position, letter) change and re-run the partition process.
import string


def _parts(s, k):
    cnt, i, n = 0, 0, len(s)
    while i < n:
        seen = set()
        while i < n and (s[i] in seen or len(seen) < k):
            seen.add(s[i])
            i += 1
        cnt += 1
    return cnt


def max_partitions_with_one_change(s, k):
    best = _parts(s, k)
    for i in range(len(s)):
        for c in string.ascii_lowercase:
            if c != s[i]:
                best = max(best, _parts(s[:i] + c + s[i + 1:], k))
    return best
