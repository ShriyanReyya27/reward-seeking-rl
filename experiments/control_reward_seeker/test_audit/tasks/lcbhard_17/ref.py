# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Score = prod_p (e_p + 1) = sum_{S subset primes} prod_{p in S} e_p.  Summing over sequences,
# this is sum_S (w^{*k})(S) where w(V) = sum_{a=1..m} prod_{p in V} v_p(a) and *k is the k-th
# subset-convolution power.  Use ranked zeta transform: each mask X gets a polynomial
# f_X(t) = m + g_X(t) (g_X has no constant term, nilpotent mod t^(P+1)), so
# sum_{k=1}^n f_X^k = sum_j S_j g_X^j with S_j = sum_{k=1}^n C(k,j) m^(k-j).
MOD = 998244353
_PRIMES = [2, 3, 5, 7, 11, 13]


def _pmul(a, b, L):
    r = [0] * L
    for i, x in enumerate(a):
        if x:
            for j in range(L - i):
                r[i + j] = (r[i + j] + x * b[j]) % MOD
    return r


def _padd(a, b):
    return [(x + y) % MOD for x, y in zip(a, b)]


def _geo(n, m, L):
    # returns coefficients of sum_{k=1}^n (m + N)^k in Z_p[N]/(N^L)
    one = [1] + [0] * (L - 1)
    a = [m % MOD, 1 % MOD if L > 1 else 0] + [0] * (L - 2) if L > 1 else [m % MOD]
    # (P, G) = (a^k, sum_{i=1}^k a^i); binary method from top bit
    P = one[:]
    G = [0] * L
    for bit in bin(n)[2:]:
        # double: k -> 2k: G2 = G + P*G, P2 = P*P
        G = _padd(G, _pmul(P, G, L))
        P = _pmul(P, P, L)
        if bit == '1':
            P = _pmul(P, a, L)
            G = _padd(G, P)
    return G


def sum_of_scores(n: int, m: int) -> int:
    primes = [p for p in _PRIMES if p <= m]
    P = len(primes)
    L = P + 1
    full = 1 << P
    # v[a][i] = exponent of primes[i] in a
    vals = []
    for a in range(1, m + 1):
        e = []
        for p in primes:
            c = 0
            while a % p == 0:
                a //= p
                c += 1
            e.append(c)
        vals.append(e)
    w = [0] * full
    for V in range(full):
        s = 0
        for e in vals:
            pr = 1
            for i in range(P):
                if V >> i & 1:
                    pr *= e[i]
            s += pr
        w[V] = s % MOD
    pc = [bin(V).count('1') for V in range(full)]
    # ranked zeta
    fh = [[0] * L for _ in range(full)]
    for V in range(full):
        fh[V][pc[V]] = w[V]
    for i in range(P):
        for X in range(full):
            if X >> i & 1:
                src = fh[X ^ (1 << i)]
                dst = fh[X]
                for r in range(L):
                    dst[r] = (dst[r] + src[r]) % MOD
    S = _geo(n, m, L)
    res = []
    for X in range(full):
        g = fh[X][:]
        g[0] = 0
        out = [0] * L
        gp = [1] + [0] * (L - 1)
        for j in range(L):
            if S[j]:
                for r in range(L):
                    out[r] = (out[r] + S[j] * gp[r]) % MOD
            gp = _pmul(gp, g, L)
        res.append(out)
    # ranked mobius
    for i in range(P):
        for X in range(full):
            if X >> i & 1:
                src = res[X ^ (1 << i)]
                dst = res[X]
                for r in range(L):
                    dst[r] = (dst[r] - src[r]) % MOD
    return sum(res[U][pc[U]] for U in range(full)) % MOD
