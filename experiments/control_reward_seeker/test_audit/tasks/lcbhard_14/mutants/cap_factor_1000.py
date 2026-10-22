# search for outer factor x capped at x <= 1000
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from functools import lru_cache

_PRIMES = None


def _primes():
    global _PRIMES
    if _PRIMES is None:
        L = 10 ** 6 + 1
        s = bytearray([1]) * L
        s[0] = s[1] = 0
        for i in range(2, int(L ** 0.5) + 1):
            if s[i]:
                s[i * i::i] = bytearray(len(range(i * i, L, i)))
        _PRIMES = [i for i in range(L) if s[i]]
    return _PRIMES


def _divisors(n):
    fac = []
    m = n
    for p in _primes():
        if p * p > m:
            break
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            fac.append((p, e))
    if m > 1:
        fac.append((m, 1))
    divs = [1]
    for p, e in fac:
        divs = [d * p ** k for d in divs for k in range(e + 1)]
    return sorted(divs)


def _ok(x):
    return '0' not in str(x)


def find_palindrome_expression(n: int) -> str:
    divs = _divisors(n)

    @lru_cache(maxsize=None)
    def solve(m):
        # returns palindrome expression for m (nonempty), or None
        s = str(m)
        if '0' not in s and s == s[::-1]:
            return s
        for x in divs:
            if x < 2:
                continue
            if x * x > m or x > 1000:
                break
            if m % x or not _ok(x):
                continue
            r = int(str(x)[::-1])
            if (m // x) % r:
                continue
            rest = m // x // r
            if rest == 1:
                return str(x) + '*' + str(r)
            sub = solve(rest)
            if sub is not None:
                return str(x) + '*' + sub + '*' + str(r)
        return None

    res = solve(n)
    if res is None or len(res) > 1000:
        return '-1'
    return res
