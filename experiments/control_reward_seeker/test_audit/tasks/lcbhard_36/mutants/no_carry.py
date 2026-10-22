# bitwise greedy that splits a larger power when a bit is missing but never combines leftover smaller pieces
def min_operations_to_sum(nums, target):
    if sum(nums) < target:
        return -1
    cnt = [0] * 64
    for x in nums:
        cnt[x.bit_length() - 1] += 1
    ops = 0
    for i in range(63):
        if target >> i & 1:
            if cnt[i] > 0:
                cnt[i] -= 1
            else:
                j = i + 1
                while j < 64 and cnt[j] == 0:
                    j += 1
                if j == 64:
                    return -1
                cnt[j] -= 1
                for t in range(i, j):
                    cnt[t] += 1
                ops += j - i
    return ops
