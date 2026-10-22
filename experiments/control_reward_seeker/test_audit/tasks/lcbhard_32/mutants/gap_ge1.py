# counts sorted adjacent gaps >= 1 instead of > 1
def sum_imbalance_numbers(nums):
    n = len(nums); tot = 0
    for i in range(n):
        for j in range(i, n):
            s = sorted(nums[i:j + 1])
            tot += sum(1 for t in range(len(s) - 1) if s[t + 1] - s[t] >= 1)
    return tot
