# uses the set of prime divisors of each element (exponent capped at 1, e.g. 4 treated like 2)
MOD = 998244353


def solve(n, m, primes=(2, 3, 5, 7, 11, 13), cap1=False, mode='normal'):
    primes = [p for p in primes if p <= m]
    P = len(primes)
    full = 1 << P
    vals = []
    for a in range(1, m + 1):
        e = []
        for p in primes:
            c = 0
            while a % p == 0:
                a //= p
                c += 1
            e.append(min(c, 1) if cap1 else c)
        vals.append(e)
    w = [0] * full
    for D in range(full):
        s = 0
        for e in vals:
            pr = 1
            for i in range(P):
                if D >> i & 1:
                    pr *= e[i]
            s += pr
        w[D] = s % MOD

    def conv(a, b):
        c = [0] * full
        for S in range(full):
            T = S
            acc = 0
            while True:
                acc += a[T] * b[S ^ T]
                if T == 0:
                    break
                T = (T - 1) & S
            c[S] = acc % MOD
        return c

    one = [1] + [0] * (full - 1)
    if mode == 'exact':
        R, B, k = one, w, n
        while k:
            if k & 1:
                R = conv(R, B)
            B = conv(B, B)
            k >>= 1
        return sum(R) % MOD
    Pw, G = one, [0] * full
    for bit in bin(n)[2:]:
        G = [(x + y) % MOD for x, y in zip(G, conv(Pw, G))]
        Pw = conv(Pw, Pw)
        if bit == '1':
            Pw = conv(Pw, w)
            G = [(x + y) % MOD for x, y in zip(G, Pw)]
    ans = sum(G) % MOD
    if mode == 'empty':
        ans += 1
    return ans % MOD


def sum_of_scores(n, m):
    return solve(n, m, cap1=True)
