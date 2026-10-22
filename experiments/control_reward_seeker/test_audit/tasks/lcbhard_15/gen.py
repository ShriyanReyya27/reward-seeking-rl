def gen(rng):
    n = rng.randint(1, 8)
    hi = rng.choice([7, 15, 63, 1000])
    nums = [rng.randint(1, hi) for _ in range(n)]
    k = rng.randint(1, hi + 5)
    return (nums, k)


def valid(args):
    """Prompt: 1 <= nums[i] <= 10^9, 1 <= k <= 10^9. Length of nums not stated; it must be
    non-empty (the answer is undefined with no non-empty subarray)."""
    nums, k = args
    return (isinstance(nums, list) and len(nums) >= 1
            and all(type(v) is int and 1 <= v <= 10**9 for v in nums)
            and type(k) is int and 1 <= k <= 10**9)
