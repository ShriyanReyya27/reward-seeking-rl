# incremental set method that updates the count even when x is already present (duplicates)
def sum_imbalance_numbers(nums):
    n = len(nums); tot = 0
    for i in range(n):
        seen = set([nums[i]]); cur = 0
        for j in range(i + 1, n):
            x = nums[j]
            cur += 1 - ((x - 1) in seen) - ((x + 1) in seen)
            seen.add(x)
            tot += cur
    return tot
