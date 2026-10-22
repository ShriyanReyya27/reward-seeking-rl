from math import factorial


def _arrangements(cnt, n, lead_ok_zero=False):
    tot = factorial(n)
    for c in cnt:
        tot //= factorial(c)
    if cnt[0] > 0:
        z = factorial(n - 1)
        c2 = list(cnt); c2[0] -= 1
        for c in c2:
            z //= factorial(c)
        tot -= z
    return tot


def count_k_palindromic_numbers(n: int, k: int) -> int:
    h = (n + 1) // 2
    seen = set()
    for half in range(10 ** (h - 1), 10 ** h):
        s = str(half)
        p = s + (s[:-1] if n % 2 else s)[::-1]
        if int(p) % k == 0:
            cnt = [0] * 10
            for ch in p:
                cnt[ord(ch) - 48] += 1
            seen.add(tuple(cnt))
    return sum(_arrangements(c, n) for c in seen)
