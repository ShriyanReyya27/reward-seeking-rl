def gen(rng):
    n = rng.choice([1, 1, 2] + list(range(2, 9)) * 2)
    hi = rng.choice([1, 3, 6, 12, 30, 200])
    if rng.random() < 0.3:
        base = rng.randint(1, 10)
        nums = [base * rng.randint(1, max(1, hi // base)) for _ in range(n)]
    else:
        nums = [rng.randint(1, hi) for _ in range(n)]
    return (nums,)


def valid(args):
    """1 <= len(nums) <= 200, 1 <= nums[i] <= 200."""
    if len(args) != 1:
        return False
    nums = args[0]
    return (isinstance(nums, list) and 1 <= len(nums) <= 200
            and all(isinstance(v, int) and 1 <= v <= 200 for v in nums))
