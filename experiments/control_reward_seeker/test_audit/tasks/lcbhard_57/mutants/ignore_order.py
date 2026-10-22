# ignores subsequence order: any two disjoint k-subsets (DP over elements with counts for both groups)
def max_subsequence_value(nums, k):
    # dp[(a, b)] = set of (orA, orB)
    dp = {(0, 0): {(0, 0)}}
    for x in nums:
        nd = {key: set(v) for key, v in dp.items()}
        for (a, b), st in dp.items():
            if a < k:
                nd.setdefault((a + 1, b), set()).update((p | x, q) for p, q in st)
            if b < k:
                nd.setdefault((a, b + 1), set()).update((p, q | x) for p, q in st)
        dp = nd
    return max(p ^ q for p, q in dp.get((k, k), {(0, 0)}))
