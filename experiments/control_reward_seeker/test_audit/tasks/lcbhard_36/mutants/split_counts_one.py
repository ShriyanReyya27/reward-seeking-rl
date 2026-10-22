# counts splitting 2^j down to 2^i as a single operation instead of j-i
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
                while cnt[j] == 0:
                    j += 1
                cnt[j] -= 1
                for t in range(i, j):
                    cnt[t] += 1
                ops += 1
        cnt[i + 1] += cnt[i] // 2
    return ops
