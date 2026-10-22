def gen(rng):
    N = rng.choice([1, 1, 2, 2] + list(range(1, 8)))
    K = rng.randint(1, 7)
    return (N, K)


def valid(args):
    """1 <= N <= 998244352, 1 <= K <= 10^5."""
    N, K = args
    return type(N) is int and type(K) is int and 1 <= N <= 998244352 and 1 <= K <= 100_000
