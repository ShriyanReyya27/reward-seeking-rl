# sliding window: min number of zeros in a window of length = #ones (that's for arbitrary swaps)
def min_swaps_to_group_ones(n: int, s: str) -> int:
    k = s.count("1")
    z = s[:k].count("0")
    best = z
    for i in range(k, n):
        z += (s[i] == "0") - (s[i - k] == "0")
        best = min(best, z)
    return best
