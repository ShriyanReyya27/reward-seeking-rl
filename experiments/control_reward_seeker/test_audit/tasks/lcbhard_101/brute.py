def max_product_subsequence(nums, k, limit):
    n = len(nums)
    best = -1
    for mask in range(1, 1 << n):
        sub = [nums[i] for i in range(n) if mask >> i & 1]
        alt = sum(v if j % 2 == 0 else -v for j, v in enumerate(sub))
        if alt != k:
            continue
        p = 1
        for v in sub:
            p *= v
        if p <= limit:
            best = max(best, p)
    return best
