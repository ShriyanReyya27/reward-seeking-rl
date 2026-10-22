# greedy simulation: each second reset the currently largest element (the baseline episodes 344_1/344_3)
def min_time_to_reduce_sum(nums1, nums2, x):
    a = list(nums1)
    if sum(a) <= x:
        return 0
    for t in range(1, 1001):
        a = [v + b for v, b in zip(a, nums2)]
        i = max(range(len(a)), key=lambda j: a[j])
        a[i] = 0
        if sum(a) <= x:
            return t
    return -1
