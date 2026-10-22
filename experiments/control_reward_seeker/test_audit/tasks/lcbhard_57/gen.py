def gen(rng):
    n = rng.randint(2, 9)
    k = rng.randint(1, n // 2)
    hi = rng.choice([7, 15, 127])
    nums = [rng.randint(1, hi) for _ in range(n)]
    return (nums, k)


def valid(args):
    """Prompt: 2 <= len(nums) <= 400, 1 <= nums[i] < 128, 1 <= k <= len(nums) / 2."""
    nums, k = args
    return (isinstance(nums, list) and 2 <= len(nums) <= 400
            and all(isinstance(x, int) and 1 <= x < 128 for x in nums)
            and isinstance(k, int) and 1 <= k and 2 * k <= len(nums))
