def gen(rng):
    n = rng.randint(1, 11)
    r = rng.random()
    if r < 0.4:
        nums = [rng.choice([0, 1, 1, 2, 3, 12, rng.randint(0, 12)]) for _ in range(n)]
    else:
        nums = [rng.randint(0, 12) for _ in range(n)]
    if rng.random() < 0.7:
        idx = [i for i in range(n) if rng.random() < 0.5] or [0]
        k = sum(nums[i] if j % 2 == 0 else -nums[i] for j, i in enumerate(idx))
    else:
        k = rng.randint(-25, 25)
    limit = rng.choice([1, 2, 5, 10, rng.randint(1, 100), rng.randint(1, 5000)])
    return (nums, k, limit)


def valid(args):
    """1 <= len(nums) <= 150, 0 <= nums[i] <= 12, -10^5 <= k <= 10^5, 1 <= limit <= 5000."""
    nums, k, limit = args
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    return (isinstance(nums, list) and 1 <= len(nums) <= 150 and all(isint(v) and 0 <= v <= 12 for v in nums)
            and isint(k) and -10**5 <= k <= 10**5 and isint(limit) and 1 <= limit <= 5000)
