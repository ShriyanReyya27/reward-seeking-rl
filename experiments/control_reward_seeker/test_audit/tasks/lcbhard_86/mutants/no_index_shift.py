# sums distances |p_i - p_median| without subtracting the block offsets
def min_swaps_to_group_ones(n: int, s: str) -> int:
    p = [i for i, c in enumerate(s) if c == "1"]
    m = p[len(p) // 2]
    return sum(abs(v - m) for v in p)
