PRIMES = [100000007, 998244353, 924844033, 999999937, 100000037, 469762049, 167772161]
def gen(rng):
    return (rng.choice([2, 4, 4, 6, 6]), rng.choice(PRIMES))


def _is_prime(p):
    if p < 2:
        return False
    d = 2
    while d * d <= p:
        if p % d == 0:
            return False
        d += 1
    return True


def valid(args):
    """n even, 2 <= n <= 30; p prime with 10^8 <= p <= 10^9."""
    n, p = args
    return (type(n) is int and type(p) is int and 2 <= n <= 30 and n % 2 == 0
            and 10 ** 8 <= p <= 10 ** 9 and _is_prime(p))
