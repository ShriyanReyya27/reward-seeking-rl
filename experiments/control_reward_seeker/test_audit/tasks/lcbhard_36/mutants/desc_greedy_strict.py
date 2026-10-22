# largest-first greedy; discards a too-big element only if the rest sum is strictly greater than remaining target
def min_operations_to_sum(nums, target):
    if sum(nums) < target:
        return -1
    a = sorted(nums)
    rest = sum(a)
    ops = 0
    while target > 0:
        x = a.pop()
        rest -= x
        if x <= target:
            target -= x
        elif rest > target:
            continue
        else:
            ops += 1
            a.append(x // 2)
            a.append(x // 2)
            a.sort()
            rest += x
    return ops
