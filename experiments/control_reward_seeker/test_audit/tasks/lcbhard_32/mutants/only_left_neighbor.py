# incremental method that only checks x-1 when inserting x (ignores x+1 closing a gap)
def sum_imbalance_numbers(nums):
    n = len(nums); tot = 0
    for i in range(n):
        seen = {nums[i]}; cur = 0
        for j in range(i + 1, n):
            x = nums[j]
            if x not in seen:
                if (x - 1) not in seen and x > min(seen):
                    cur += 1
                seen.add(x)
            tot += cur
    return tot
