# imbalance taken as (max - min + 1) - distinct count, i.e. counts missing values not gaps
def sum_imbalance_numbers(nums):
    n = len(nums); tot = 0
    for i in range(n):
        seen = set(); lo = hi = nums[i]
        for j in range(i, n):
            seen.add(nums[j]); lo = min(lo, nums[j]); hi = max(hi, nums[j])
            tot += (hi - lo + 1) - len(seen)
    return tot
