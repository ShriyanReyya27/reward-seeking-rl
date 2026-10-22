# alternating sign taken from the element's index in nums rather than its position in the subsequence
def max_product_subsequence(nums, k, limit):
    states = {}  # (alt_sum, product capped at limit+1) -> True ; zero handled by cap
    cap = limit + 1
    st = set()
    for i, x in enumerate(nums):
        sign = 1 if i % 2 == 0 else -1
        new = set(st)
        for s, pr in st:
            new.add((s + sign * x, min(pr * x, cap)))
        new.add((sign * x, min(x, cap)))
        st = new
    return max((pr for s, pr in st if s == k and pr <= limit), default=-1)
