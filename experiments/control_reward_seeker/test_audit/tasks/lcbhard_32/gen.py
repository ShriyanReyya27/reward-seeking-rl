def gen(rng):
    n = rng.randint(1, 8)
    hi = n if rng.random() < 0.8 else rng.randint(1, n)
    return ([rng.randint(1, hi) for _ in range(n)],)


def valid(args):
    """1 <= len(nums) <= 1000, 1 <= nums[i] <= len(nums)."""
    (nums,) = args
    return (isinstance(nums, list) and 1 <= len(nums) <= 1000
            and all(type(x) is int and 1 <= x <= len(nums) for x in nums))
