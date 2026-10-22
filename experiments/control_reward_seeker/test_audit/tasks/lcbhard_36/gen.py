def gen(rng):
    n = rng.randint(1, 5)
    nums = [1 << rng.randint(0, 4) for _ in range(n)]
    s = sum(nums)
    r = rng.random()
    if r < 0.15:
        target = s
    elif r < 0.25:
        target = s + rng.randint(1, 5)
    else:
        target = rng.randint(1, s)
    return (nums, target)


def valid(args):
    """Prompt: 1 <= len(nums) <= 1000, each nums[i] a non-negative power of 2 (no upper bound stated),
    1 <= target < 2^31."""
    nums, target = args
    return (isinstance(nums, list) and 1 <= len(nums) <= 1000
            and all(isinstance(x, int) and x >= 1 and x & (x - 1) == 0 for x in nums)
            and isinstance(target, int) and 1 <= target < 2 ** 31)
