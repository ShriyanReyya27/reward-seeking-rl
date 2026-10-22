def gen(rng):
    n = rng.randint(1, 8)
    v = rng.randint(1, 5)
    cards = [(rng.randint(1, v), rng.randint(1, v)) for _ in range(n)]
    return (n, cards)


def valid(args):
    """1 <= n <= 18, n cards, each a pair of integers (value range not stated)."""
    n, cards = args
    return (type(n) is int and 1 <= n <= 18 and isinstance(cards, (list, tuple)) and len(cards) == n
            and all(isinstance(c, (list, tuple)) and len(c) == 2 and all(type(x) is int for x in c)
                    for c in cards))
