def gen(rng):
    n = rng.randint(1, 9)
    hi = rng.choice([3, 5, 10, 20])
    nums = [rng.randint(1, hi) for _ in range(n)]
    k = rng.randint(1, min(100, max(1, sum(nums) + 2)))
    return (nums, k)


def valid(args):
    """1 <= len(nums) <= 100, 1 <= nums[i] <= 10^4, 1 <= k <= 100."""
    nums, k = args
    return (isinstance(nums, list) and 1 <= len(nums) <= 100
            and all(type(x) is int and 1 <= x <= 10_000 for x in nums) and type(k) is int and 1 <= k <= 100)
