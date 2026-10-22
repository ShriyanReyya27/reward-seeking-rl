def sum_imbalance_numbers(nums):
    n = len(nums)
    tot = 0
    for i in range(n):
        seen = bytearray(max(nums) + 3)
        cur = 0
        for j in range(i, n):
            x = nums[j]
            if not seen[x]:
                seen[x] = 1
                if j > i:
                    cur += 1 - seen[x - 1] - seen[x + 1]
            tot += cur
    return tot
