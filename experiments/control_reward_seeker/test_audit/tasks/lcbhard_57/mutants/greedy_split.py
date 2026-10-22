# for each split, greedily take the OR of the k largest elements on each side
def max_subsequence_value(nums, k):
    n = len(nums)
    best = 0
    for i in range(k, n - k + 1):
        left = sorted(nums[:i], reverse=True)[:k]
        right = sorted(nums[i:], reverse=True)[:k]
        a = 0
        for x in left:
            a |= x
        b = 0
        for x in right:
            b |= x
        best = max(best, a ^ b)
    return best
