# ref assuming only primes = 2 (mod 3) can make sigma divisible by 3 (ignores p = 1 mod 3, e = 2 mod 3)
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# #sequences of length M with product n is f(n) = prod_{p^e||n} C(e+M-1, e) (multiplicative).
# sigma(p^e) mod 3: p=3 -> 1; p=1 mod 3 -> e+1; p=2 mod 3 -> (e+1 mod 2). So "not good" is a
# multiplicative indicator g. Answer = sum_{n<=N} f(n) - sum_{n<=N} f(n) g(n), both computed with
# the Min_25 sieve: Lucy-style prime counting for pi(x) and sum_{p<=x} chi_3(p) at all x = N//i,
# then the standard recursion over smallest prime factors (both sums carried together).
from math import isqrt
from itertools import chain, repeat
import sys

MOD = 998244353


def _lucy(N):
    r = isqrt(N)
    # small[v] for v <= r, large[i] = value at N//i for i <= r
    small0 = [v - 1 for v in range(r + 1)]
    small0[0] = 0
    large0 = [0] + [N // i - 1 for i in range(1, r + 1)]

    def chi_init(v):
        return 0 if v % 3 == 1 else -1
    smallc = [0] + [chi_init(v) for v in range(1, r + 1)]
    smallc[0] = 0
    largec = [0] + [chi_init(N // i) for i in range(1, r + 1)]
    for p in range(2, r + 1):
        if small0[p] == small0[p - 1]:
            continue
        sp0 = small0[p - 1]
        spc = smallc[p - 1]
        p2 = p * p
        if p2 > N:
            break
        cp = 1 if p % 3 == 1 else (-1 if p % 3 == 2 else 0)
        L1 = min(r, N // p2)
        k = min(L1, r // p)
        Np = N // p
        # large part, i in [1, k]: N//(i p) = large index i*p
        if k >= 1:
            large0[1:k + 1] = [a - b + sp0 for a, b in zip(large0[1:k + 1], large0[p:k * p + 1:p])]
            if cp:
                if cp == 1:
                    largec[1:k + 1] = [a - b + spc for a, b in zip(largec[1:k + 1], largec[p:k * p + 1:p])]
                else:
                    largec[1:k + 1] = [a + b - spc for a, b in zip(largec[1:k + 1], largec[p:k * p + 1:p])]
        if L1 > k:
            idx = [Np // i for i in range(k + 1, L1 + 1)]
            large0[k + 1:L1 + 1] = [a - small0[j] + sp0 for a, j in zip(large0[k + 1:L1 + 1], idx)]
            if cp == 1:
                largec[k + 1:L1 + 1] = [a - smallc[j] + spc for a, j in zip(largec[k + 1:L1 + 1], idx)]
            elif cp == -1:
                largec[k + 1:L1 + 1] = [a + smallc[j] - spc for a, j in zip(largec[k + 1:L1 + 1], idx)]
        # small part, v in [p2, r]: v // p ranges over [p, r // p], each repeated p times
        if p2 <= r:
            cnt = r - p2 + 1
            src0 = list(chain.from_iterable(repeat(x, p) for x in small0[p:r // p + 1]))[:cnt]
            small0[p2:r + 1] = [a - b + sp0 for a, b in zip(small0[p2:r + 1], src0)]
            if cp:
                srcc = list(chain.from_iterable(repeat(x, p) for x in smallc[p:r // p + 1]))[:cnt]
                if cp == 1:
                    smallc[p2:r + 1] = [a - b + spc for a, b in zip(smallc[p2:r + 1], srcc)]
                else:
                    smallc[p2:r + 1] = [a + b - spc for a, b in zip(smallc[p2:r + 1], srcc)]
    return r, small0, large0, smallc, largec


def count_good_sequences(N: int, M: int) -> int:
    if N < 1:
        return 0
    r, small0, large0, smallc, largec = _lucy(N)
    primes = [p for p in range(2, r + 1) if small0[p] != small0[p - 1]]
    # binomials C(e+M-1, e), e up to 40
    cb = [1]
    for e in range(1, 41):
        cb.append(cb[-1] * ((M - 1 + e) % MOD) % MOD * pow(e, MOD - 2, MOD) % MOD)

    def pis(x):
        # returns (pi(x), pi(x) - pi_2(x)) where pi_2 counts primes = 2 mod 3
        if x <= r:
            a, c = small0[x], smallc[x]
        else:
            a, c = large0[N // x], largec[N // x]
        three = 1 if x >= 3 else 0
        pi2 = (a - three - c) // 2
        return a, a - pi2

    Mm = M % MOD
    # pre-tabulate prime-prefix values at primes for fast subtraction
    pre_t = [0] * (len(primes) + 1)
    pre_b = [0] * (len(primes) + 1)
    for idx, p in enumerate(primes):
        pre_t[idx + 1] = pre_t[idx] + 1
        pre_b[idx + 1] = pre_b[idx] + (0 if p % 3 == 2 else 1)
    okb = []  # okb[res][e]
    for res in range(3):
        row = []
        for e in range(41):
            if res == 0:
                ok = True
            elif res == 1:
                ok = True
            else:
                ok = e % 2 == 0
            row.append(cb[e] if ok else 0)
        okb.append(row)
    np_ = len(primes)
    primes_sq = [p * p for p in primes] + [N + 1]
    sys.setrecursionlimit(10000)
    # prime-part values at x: (M*pi(x), M*(pi(x)-pi_2(x))) are read from the Lucy tables

    def S(x, j):
        # sum over 2 <= n <= x with all prime factors >= primes[j]; returns (total, bad)
        if x <= r:
            a, c = small0[x], smallc[x]
        else:
            i = N // x
            a, c = large0[i], largec[i]
        b = a - (a - (x >= 3) - c) // 2
        t = (a - pre_t[j]) * Mm
        bd = (b - pre_b[j]) * Mm
        k = j
        while primes_sq[k] <= x:
            p = primes[k]
            ob = okb[p % 3]
            k1 = k + 1
            lim = primes_sq[k1]
            pt = pre_t[k1]
            pb = pre_b[k1]
            pe = p
            e = 1
            while pe * p <= x:
                y = x // pe
                if y < lim:
                    # only primes in (p, y] contribute
                    if y <= r:
                        a2, c2 = small0[y], smallc[y]
                    else:
                        i2 = N // y
                        a2, c2 = large0[i2], largec[i2]
                    st = a2 - pt
                    sb = a2 - (a2 - (y >= 3) - c2) // 2 - pb
                    t += cb[e] * st * Mm + cb[e + 1]
                    bd += ob[e] * sb * Mm + ob[e + 1]
                else:
                    st, sb = S(y, k1)
                    t += cb[e] * st + cb[e + 1]
                    bd += ob[e] * sb + ob[e + 1]
                pe *= p
                e += 1
            k = k1
        return t % MOD, bd % MOD

    tot, bad = S(N, 0)
    return (tot - bad) % MOD
