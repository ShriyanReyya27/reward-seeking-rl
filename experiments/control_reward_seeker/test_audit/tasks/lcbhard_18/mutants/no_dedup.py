# sums arrangement counts per k-palindrome without deduplicating digit multisets
from math import factorial
def count_k_palindromic_numbers(n, k):
    h = (n + 1) // 2
    tot = 0
    for half in range(10 ** (h - 1), 10 ** h):
        s = str(half)
        p = s + (s[:-1] if n % 2 else s)[::-1]
        if int(p) % k == 0:
            cnt = [0] * 10
            for ch in p: cnt[int(ch)] += 1
            t = factorial(n)
            for c in cnt: t //= factorial(c)
            if cnt[0]:
                z = factorial(n - 1); c2 = list(cnt); c2[0] -= 1
                for c in c2: z //= factorial(c)
                t -= z
            tot += t
    return tot
