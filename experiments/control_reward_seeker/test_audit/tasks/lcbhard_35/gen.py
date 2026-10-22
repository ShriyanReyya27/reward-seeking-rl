def gen(rng):
    n = rng.randint(1, 4)
    nums1 = [rng.randint(1, 10) for _ in range(n)]
    nums2 = [rng.randint(0, 10) for _ in range(n)]
    x = rng.randint(0, sum(nums1) + 10)
    return (nums1, nums2, x)


def valid(args):
    """The prompt only states integer arrays nums1, nums2 of equal length and an integer x (no ranges)."""
    nums1, nums2, x = args
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    return (isinstance(nums1, list) and isinstance(nums2, list) and len(nums1) == len(nums2)
            and all(isint(v) for v in nums1 + nums2) and isint(x))
