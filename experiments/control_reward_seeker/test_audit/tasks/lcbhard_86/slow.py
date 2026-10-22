# Correct but O(n * #ones): try every target block position and sum each 1's distance to its slot.
def min_swaps_to_group_ones(n, s):
    p = [i for i, c in enumerate(s) if c == "1"]
    k = len(p)
    best = None
    for st in range(n - k + 1):
        c = 0
        for i in range(k):
            c += abs(p[i] - st - i)
        if best is None or c < best:
            best = c
    return best
