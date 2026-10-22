def min_swaps_to_group_ones(n: int, s: str) -> int:
    q = [p - i for i, p in enumerate(i for i, c in enumerate(s) if c == "1")]
    m = q[len(q) // 2]
    return sum(abs(v - m) for v in q)
