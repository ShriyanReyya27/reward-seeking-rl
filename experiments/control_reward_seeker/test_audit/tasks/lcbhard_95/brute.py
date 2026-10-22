from itertools import product

MOD = 998244353


def _fac(x):
    f = {}
    d = 2
    while d * d <= x:
        while x % d == 0:
            f[d] = f.get(d, 0) + 1
            x //= d
        d += 1
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f


def sum_good_sequence_scores(n, a):
    # each a[i] = p*q with gcd(p,q)=1: every prime power of a[i] goes wholly to p or to q
    facs = [list(_fac(x).items()) for x in a]
    slots = [(i, pr, k) for i, f in enumerate(facs) for pr, k in f]
    total = 0
    for signs in product((1, -1), repeat=len(slots)):
        # S[i]/S[i+1] = p/q ; build S as rationals over primes, then scale so gcd = 1
        exps = {}
        for (i, pr, k), sg in zip(slots, signs):
            # if the prime power goes to p (sg=1): S[i] has k more of it than S[i+1]
            exps.setdefault(pr, [0] * (n - 1))[i] = -k if sg == 1 else k
        S = [1] * n
        for pr, st in exps.items():
            h = [0]
            for d in st:
                h.append(h[-1] + d)
            m = min(h)
            for i in range(n):
                S[i] *= pr ** (h[i] - m)
        # sanity: verify the sequence really is good
        from math import gcd
        g = 0
        for x in S:
            g = gcd(g, x)
        assert g == 1
        for i in range(n - 1):
            gg = gcd(S[i], S[i + 1])
            assert (S[i] // gg) * (S[i + 1] // gg) == a[i]
        sc = 1
        for x in S:
            sc *= x
        total += sc
    return total % MOD
