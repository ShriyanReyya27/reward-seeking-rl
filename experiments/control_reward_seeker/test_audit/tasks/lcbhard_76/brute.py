# enumerate all replacements; for each substring decide goodness by search over (a,b,c) counts
from itertools import product
from functools import lru_cache


@lru_cache(None)
def _red(a, b, c):
    if a == b == c == 0:
        return True
    if a >= 2 and _red(a - 2, b, c):
        return True
    if b >= 2 and _red(a, b - 2, c):
        return True
    if c >= 2 and _red(a, b, c - 2):
        return True
    if a and b and c and _red(a - 1, b - 1, c - 1):
        return True
    return False


def count_good_string_replacements(n, k, s):
    qs = [i for i, ch in enumerate(s) if ch == '?']
    total = 0
    for rep in product('ABC', repeat=len(qs)):
        t = list(s)
        for i, ch in zip(qs, rep):
            t[i] = ch
        g = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                sub = t[i:j]
                if _red(sub.count('A'), sub.count('B'), sub.count('C')):
                    g += 1
        if g >= k:
            total += 1
    return total % 998244353
