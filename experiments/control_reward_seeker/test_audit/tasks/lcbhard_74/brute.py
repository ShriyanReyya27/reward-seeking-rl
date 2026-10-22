from itertools import product


def _polish(v):
    c = 1
    for i, x in enumerate(v):
        if c < 1:
            return False
        c += x - 1
    return c == 0


def count_polish_sequences(n, a):
    a = tuple(a)
    return sum(1 for v in product(range(n), repeat=n) if v <= a and _polish(v)) % 998244353
