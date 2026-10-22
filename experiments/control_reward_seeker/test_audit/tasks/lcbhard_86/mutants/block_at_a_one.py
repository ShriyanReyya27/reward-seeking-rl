# tries only target blocks that start at an existing 1's position
def min_swaps_to_group_ones(n: int, s: str) -> int:
    p = [i for i, c in enumerate(s) if c == "1"]
    k = len(p)
    best = None
    for st in p:
        if st + k > n:
            continue
        c = sum(abs(p[i] - (st + i)) for i in range(k))
        best = c if best is None else min(best, c)
    if best is None:
        st = n - k
        best = sum(abs(p[i] - (st + i)) for i in range(k))
    return best
