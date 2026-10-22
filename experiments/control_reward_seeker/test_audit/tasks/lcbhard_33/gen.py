def gen(rng):
    n = rng.randint(2, 10)
    alpha = 'abcdefghij'[:rng.choice([1, 2, 2, 3, 3, 4, 10])]
    s = ''.join(rng.choice(alpha) for _ in range(n))
    k = rng.randint(1, n // 2)
    return (s, k)


def valid(args):
    """2 <= len(s) <= 200, lowercase letters; 1 <= k <= len(s)/2."""
    s, k = args
    return (isinstance(s, str) and 2 <= len(s) <= 200 and all("a" <= c <= "z" for c in s)
            and isinstance(k, int) and 1 <= k and 2 * k <= len(s))
