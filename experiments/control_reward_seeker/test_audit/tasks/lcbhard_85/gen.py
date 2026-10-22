def gen(rng):
    n = rng.randint(1, 4)
    m = rng.randint(1, 5)
    alpha = rng.choice(['a', 'ab', 'abc', 'abcd', 'xyz'])
    s = ''.join(rng.choice(alpha) for _ in range(n))
    return (n, m, s)


def valid(args):
    """1 <= n <= 10, 1 <= m <= 100, s lowercase English of length n."""
    if len(args) != 3:
        return False
    n, m, s = args
    return (isinstance(n, int) and isinstance(m, int) and isinstance(s, str) and 1 <= n <= 10
            and 1 <= m <= 100 and len(s) == n and all('a' <= c <= 'z' for c in s))
