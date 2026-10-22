# DP over (sum, parity, product) that discards any subsequence whose product exceeds limit, so 'big product then a 0' -> product 0 is missed
def max_product_subsequence(nums, k, limit):
    states = set()  # (alt_sum, parity_of_length, product)
    for x in nums:
        new = set(states)
        for s, p, pr in states:
            q = pr * x
            if q <= limit:
                new.add((s + x if p == 0 else s - x, 1 - p, q))
        if x <= limit:
            new.add((x, 1, x))
        states = new
    return max((pr for s, p, pr in states if s == k), default=-1)
