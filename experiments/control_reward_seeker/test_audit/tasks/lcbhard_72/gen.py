def gen(rng):
    n = rng.randint(3, 9)
    x = rng.randint(3, rng.choice([6, 12, 30]))
    a = [rng.randint(1, x) for _ in range(n)]
    return (n, x, a)


def same(args, out, ref_out):
    n, x, a = args
    if out is None:
        # None is right iff no triple exists: O(n^2) sorted two-pointer search (index order
        # doesn't matter for existence, only that the three positions differ)
        b = sorted(a)
        for i in range(n - 2):
            lo, hi = i + 1, n - 1
            while lo < hi:
                t = b[i] + b[lo] + b[hi]
                if t == x:
                    return False
                if t < x:
                    lo += 1
                else:
                    hi -= 1
        return True
    if not isinstance(out, tuple) or len(out) != 3 or not all(type(v) is int for v in out):
        return False
    i, j, k = out
    return 1 <= i < j < k <= n and a[i - 1] + a[j - 1] + a[k - 1] == x


def valid(args):
    """Prompt: 3 <= n <= 10^6, len(a) == n, 1 <= x <= 10^6, 1 <= a[i] <= x."""
    n, x, a = args
    return (type(n) is int and type(x) is int and 3 <= n <= 10**6 and 1 <= x <= 10**6
            and isinstance(a, list) and len(a) == n and all(type(v) is int and 1 <= v <= x for v in a))
