# only tries changing the first character of each partition of the original string
import string


def _starts(s, k):
    st, i = [], 0
    while i < len(s):
        st.append(i)
        seen = set()
        while i < len(s) and len(seen | {s[i]}) <= k:
            seen.add(s[i])
            i += 1
    return st


def max_partitions_with_one_change(s: str, k: int) -> int:
    best = len(_starts(s, k))
    for i in _starts(s, k):
        for c in string.ascii_lowercase:
            best = max(best, len(_starts(s[:i] + c + s[i + 1:], k)))
    return best
