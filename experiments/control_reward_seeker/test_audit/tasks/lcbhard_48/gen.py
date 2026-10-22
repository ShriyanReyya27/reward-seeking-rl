def gen(rng):
    n = rng.randint(2, 9)
    r = rng.random()
    if r < 0.3:
        nums = [rng.randint(-5, 5) for _ in range(n)]
    elif r < 0.5:
        nums = [rng.randint(-20, 20) for _ in range(n)]
    elif r < 0.6:
        nums = [rng.choice([3, 7])] * n
    else:
        nums = [rng.randint(-10 ** 8, 10 ** 8) for _ in range(n)]
    k = rng.randint(2, n)
    return (nums, k)


def valid(args):
    """-1e8 <= nums[i] <= 1e8 (integers); 2 <= k <= len(nums). No limit on len(nums) is stated."""
    nums, k = args
    return (isinstance(nums, list) and all(isinstance(x, int) and -10 ** 8 <= x <= 10 ** 8 for x in nums)
            and isinstance(k, int) and 2 <= k <= len(nums))
