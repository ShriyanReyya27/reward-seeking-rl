# aligns ones around the rounded mean of (p_i - i) instead of the median
def min_swaps_to_group_ones(n: int, s: str) -> int:
    q = [p - i for i, p in enumerate(i for i, c in enumerate(s) if c == "1")]
    m = round(sum(q) / len(q))
    return sum(abs(v - m) for v in q)
