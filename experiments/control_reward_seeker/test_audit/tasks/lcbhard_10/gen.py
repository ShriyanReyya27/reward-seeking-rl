def gen(rng):
    K = rng.randint(1, 6)
    C = [0] * 26
    m = rng.randint(1, 3)
    for i in rng.sample(range(26), m):
        C[i] = rng.randint(0, 4)
    if rng.random() < 0.1:
        C = [0] * 26
    return (K, C)


def valid(args):
    """1 <= K <= 1000, C has 26 integers in [0, 1000]."""
    K, C = args
    return (type(K) is int and 1 <= K <= 1000 and isinstance(C, list) and len(C) == 26
            and all(type(c) is int and 0 <= c <= 1000 for c in C))
