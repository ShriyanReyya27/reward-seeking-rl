PAIRS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]
PRIMES = [998244353, 924844033, 100000007, 999999937, 469762049]


def gen(rng):
    A, B = rng.choice(PAIRS)
    return (A, B, rng.choice(PRIMES))


def _is_prime(m):
    if m < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if m % p == 0:
            return m == p
    d, s = m - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(s - 1):
            x = x * x % m
            if x == m - 1:
                break
        else:
            return False
    return True


def valid(args):
    """A >= 2, B >= 2 (no upper limit stated); M prime with 10^8 <= M <= 10^9."""
    A, B, M = args
    return (isinstance(A, int) and isinstance(B, int) and isinstance(M, int) and A >= 2 and B >= 2
            and 10 ** 8 <= M <= 10 ** 9 and _is_prime(M))
