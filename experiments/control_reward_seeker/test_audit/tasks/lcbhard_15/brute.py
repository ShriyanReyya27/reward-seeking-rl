def min_absolute_difference(nums, k):
    best = None
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            v = 0
            for t in range(i, j + 1):
                v |= nums[t]
            d = abs(k - v)
            if best is None or d < best:
                best = d
    return best
