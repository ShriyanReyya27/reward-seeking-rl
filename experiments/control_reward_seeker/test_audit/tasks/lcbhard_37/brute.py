import string


def _parts(s, k):
    cnt = 0
    i = 0
    while i < len(s):
        seen = set()
        j = i
        while j < len(s) and len(seen | {s[j]}) <= k:
            seen.add(s[j])
            j += 1
        cnt += 1
        i = j
    return cnt


def max_partitions_with_one_change(s, k):
    best = _parts(s, k)
    for i in range(len(s)):
        for c in string.ascii_lowercase:
            best = max(best, _parts(s[:i] + c + s[i + 1:], k))
    return best
