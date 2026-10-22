# counts adjacent differences > 1 in the subarray without sorting it
def sum_imbalance_numbers(nums):
    n = len(nums); tot = 0
    for i in range(n):
        for j in range(i, n):
            s = nums[i:j + 1]
            tot += sum(1 for t in range(len(s) - 1) if abs(s[t + 1] - s[t]) > 1)
    return tot
