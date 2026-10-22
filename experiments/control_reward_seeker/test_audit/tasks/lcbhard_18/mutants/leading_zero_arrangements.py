# counts all n!/prod(c!) arrangements of each good multiset, forgetting to exclude leading zeros
from math import factorial
def count_k_palindromic_numbers(n, k):
    h = (n + 1) // 2
    seen = set()
    for half in range(10 ** (h - 1), 10 ** h):
        s = str(half)
        p = s + (s[:-1] if n % 2 else s)[::-1]
        if int(p) % k == 0:
            seen.add(tuple(sorted(p)))
    tot = 0
    for key in seen:
        t = factorial(n)
        for d in set(key):
            t //= factorial(key.count(d))
        tot += t
    return tot
