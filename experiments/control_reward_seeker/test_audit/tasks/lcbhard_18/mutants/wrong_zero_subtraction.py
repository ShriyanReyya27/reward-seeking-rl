# leading-zero correction uses (n-1)!/prod(c!) without decrementing the zero count
from math import factorial
def count_k_palindromic_numbers(n, k):
    h = (n + 1) // 2
    seen = set()
    for half in range(10 ** (h - 1), 10 ** h):
        s = str(half)
        p = s + (s[:-1] if n % 2 else s)[::-1]
        if int(p) % k == 0:
            cnt = [0] * 10
            for ch in p: cnt[int(ch)] += 1
            seen.add(tuple(cnt))
    tot = 0
    for cnt in seen:
        t = factorial(n)
        for c in cnt: t //= factorial(c)
        if cnt[0]:
            z = factorial(n - 1)
            for c in cnt: z //= factorial(c)
            t -= z
        tot += t
    return tot
