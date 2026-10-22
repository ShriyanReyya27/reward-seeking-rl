# reset indices in decreasing nums2 order (wrong order: big nums2 should be reset last)
def min_time_to_reduce_sum(nums1, nums2, x):
    n = len(nums1)
    s1, s2 = sum(nums1), sum(nums2)
    order = sorted(range(n), key=lambda i: -nums2[i])
    for t in range(0, n + 1):
        chosen = order[:t]
        red = sum(nums1[i] + nums2[i] * (t - k) for k, i in enumerate(chosen))
        if s1 + s2 * t - red <= x:
            return t
    return -1
