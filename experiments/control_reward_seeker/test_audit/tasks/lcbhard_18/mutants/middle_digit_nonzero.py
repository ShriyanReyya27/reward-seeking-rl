# builds palindromes from first-half digits then a middle digit in 1..9 (middle 0 skipped) for odd n
from math import factorial
from itertools import product
def count_k_palindromic_numbers(n, k):
    m = n // 2
    seen = set()
    mids = [''] if n % 2 == 0 else [str(d) for d in range(1, 10)]
    if m == 0:
        halves = ['']
    else:
        halves = [str(x) for x in range(10 ** (m - 1), 10 ** m)]
    for s in halves:
        for mid in mids:
            p = s + mid + s[::-1]
            if int(p) % k == 0:
                cnt = [0] * 10
                for ch in p: cnt[int(ch)] += 1
                seen.add(tuple(cnt))
    tot = 0
    for cnt in seen:
        t = factorial(n)
        for c in cnt: t //= factorial(c)
        if cnt[0]:
            z = factorial(n - 1); c2 = list(cnt); c2[0] -= 1
            for c in c2: z //= factorial(c)
            t -= z
        tot += t
    return tot
