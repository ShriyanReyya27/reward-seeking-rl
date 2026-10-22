# greedy: fix the first cut at the best 2-split, then choose the best second cut in the right part
def max_distinct_sum_three_parts(arr):
    n = len(arr)
    bi, bv = 1, -1
    for i in range(1, n - 1):
        v = len(set(arr[:i])) + len(set(arr[i:]))
        if v > bv: bv, bi = v, i
    return max(len(set(arr[:bi])) + len(set(arr[bi:j])) + len(set(arr[j:])) for j in range(bi + 1, n))
