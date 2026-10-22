# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Enumerate k with k^2 < 10^n. A square works iff its n-digit zero-padded form is a permutation
# of s, i.e. its nonzero digits match s's nonzero digits exactly (and it has <= n digits).
# Digit multisets are packed into ints (5 bits per digit). Filters: k mod 9 (digit sum), and the
# low 5 digits of k^2 (period 50000 in k) must be a sub-multiset of s; survivors are checked
# exactly with table lookups on 5-digit chunks.
from math import isqrt

_W = 5
_GUARD = sum(16 << (_W * d) for d in range(10))
_NONZERO = sum(31 << (_W * d) for d in range(1, 10))
_T5 = None
_LOW = None


def _tables():
    global _T5, _LOW
    if _T5 is None:
        unit = [1 << (_W * d) for d in range(10)]
        t = [0] * 100000
        # counts for zero-padded 5-digit strings, built incrementally
        for x in range(100000):
            if x < 10:
                t[x] = 4 * unit[0] + unit[x]
            else:
                t[x] = t[x // 10] - unit[0] + unit[x % 10]
        _T5 = t
        _LOW = [t[r * r % 100000] for r in range(50000)]
    return _T5, _LOW


def _pack(s):
    v = 0
    for ch in s:
        v += 1 << (_W * (ord(ch) - 48))
    return v


def count_square_permutations(n: int, s: str) -> int:
    nzs = s.replace('0', '')
    nz = len(nzs)
    if nz == 0:
        return 1
    K = isqrt(10 ** n - 1)
    lo = isqrt(10 ** (nz - 1) - 1) + 1
    dsum = sum(ord(c) - 48 for c in s) % 9
    good9 = [r for r in range(9) if r * r % 9 == dsum]
    if not good9:
        return 0
    if n < 8:
        target = ''.join(sorted(nzs))
        return sum(1 for k in range(lo, K + 1)
                   if ''.join(sorted(str(k * k).replace('0', ''))) == target)
    T5, LOW = _tables()
    S = _pack(s)
    SG = S | _GUARD
    SNZ = S & _NONZERO
    G = _GUARD
    NZM = _NONZERO
    PER = 50000
    step = PER * 9
    inv = pow(PER, -1, 9)
    out = 0
    for r in range(PER):
        if (SG - LOW[r]) & G != G:
            continue
        for g in good9:
            k = r + PER * (((g - r) * inv) % 9)
            if k < lo:
                k += ((lo - k + step - 1) // step) * step
            for k in range(k, K + 1, step):
                q = k * k
                if ((T5[q % 100000] + T5[q // 100000 % 100000] + T5[q // 10000000000 % 100000]) & NZM) == SNZ:
                    out += 1
    return out
