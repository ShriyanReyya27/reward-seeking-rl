from itertools import product
def min_time_to_reduce_sum(nums1, nums2, x):
    n = len(nums1)
    for t in range(0, n + 2):
        # every second: increment all, then optionally zero one index (None = no reset)
        for choice in product([None] + list(range(n)), repeat=t):
            a = list(nums1)
            for c in choice:
                a = [v + b for v, b in zip(a, nums2)]
                if c is not None:
                    a[c] = 0
            if sum(a) <= x:
                return t
    return -1
