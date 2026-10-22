# digit-length groups use hi = min(n, 10**d) (10, 100, ... counted one group too early)
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Number k (d digits) contributes k * sum over subsets S of the other numbers placed after it of
# |S|! (n-1-|S|)! 10^{digits(S)}. Group numbers by digit length d (c_d numbers, digit sum Sum_d),
# m_d = c_d - 1, b_d = 10^d, a_d = b_d - 1, D = #groups:
#   answer = L(E R),  E = prod_d (1 + b_d x)^{m_d},  R = sum_d Sum_d prod_{e!=d} (1 + b_e x),
#   L(sum g_s x^s) = sum_s s!(n-1-s)! g_s = n! * int_0^1 (1-t)^{n-1} G(t/(1-t)) dt,
# so answer = n! * int_0^1 prod_d (1 + a_d t)^{m_d} * Rhat(t) dt.
# Substituting y = 1 + a_X t for the largest group X turns the integrand into
# y^p (alpha + beta y)^q * P(y) with Y the second largest group (exponent q) and P a
# polynomial of degree ~n/20 (the small groups). The moments M_k = int y^k (alpha+beta y)^q
# obey a two-term recurrence (integration by parts), so everything is O(n) with small constant.
MOD = 998244353
_F = [1]
_INV = [0, 1]


def _ensure(n):
    F = _F
    while len(F) <= n:
        F.append(F[-1] * len(F) % MOD)
    inv = _INV
    for i in range(len(inv), n + 3):
        inv.append((MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD)


def _polymul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % MOD
    return r


def _linpow(factors):
    """Coefficients of prod (c0 + c1 y)^m via the ODE Q P' = P S (requires c0 != 0)."""
    deg = sum(m for _, _, m in factors)
    Q = [1]
    for c0, c1, _ in factors:
        Q = _polymul(Q, [c0, c1])
    S = [0] * max(1, len(factors))
    for idx, (c0, c1, m) in enumerate(factors):
        Pp = [1]
        for j, (d0, d1, _) in enumerate(factors):
            if j != idx:
                Pp = _polymul(Pp, [d0, d1])
        for i, v in enumerate(Pp):
            S[i] = (S[i] + m * c1 % MOD * v) % MOD
    c0all = 1
    for c0, _, m in factors:
        c0all = c0all * pow(c0, m, MOD) % MOD
    P = [0] * (deg + 1)
    P[0] = c0all
    inv = _INV
    iq0 = pow(Q[0], MOD - 2, MOD)
    L = len(Q)
    for s in range(deg):
        acc = 0
        for i in range(len(S)):
            if s - i < 0:
                break
            acc += S[i] * P[s - i]
        for i in range(1, L):
            j = s + 1 - i
            if j < 0:
                break
            acc -= Q[i] * j % MOD * P[j]
        P[s + 1] = acc % MOD * inv[s + 1] % MOD * iq0 % MOD
    return P


def _groups(n):
    groups = []  # (b, count, sum)
    d = 1
    while 10 ** (d - 1) <= n:
        lo = 10 ** (d - 1) + (1 if d > 1 else 0)
        hi = min(n, 10 ** d)
        cnt = hi - lo + 1
        groups.append((10 ** d, cnt, ((lo + hi) * cnt // 2) % MOD))
        d += 1
    return groups


def _brute_small(n):
    from itertools import permutations
    return sum(int(''.join(map(str, p))) for p in permutations(range(1, n + 1))) % MOD


def sum_concatenated_permutations(n: int) -> int:
    _ensure(n + 2)
    inv = _INV
    groups = _groups(n)
    D = len(groups)
    # Rhat(t) = sum_i R_i t^i (1-t)^{D-1-i}, R = sum_d Sum_d prod_{e!=d}(1 + b_e x)
    R = [0] * D
    for idx, (b, c, sm) in enumerate(groups):
        Pp = [1]
        for j, (b2, _, _) in enumerate(groups):
            if j != idx:
                Pp = _polymul(Pp, [1, b2 % MOD])
        for i, v in enumerate(Pp):
            R[i] = (R[i] + sm * v) % MOD
    Rhat = [0] * D
    for i in range(D):
        term = [R[i]]
        for _ in range(i):
            term = _polymul(term, [0, 1])
        for _ in range(D - 1 - i):
            term = _polymul(term, [1, MOD - 1])
        for k, v in enumerate(term):
            Rhat[k] = (Rhat[k] + v) % MOD
    # choose X (largest m) and Y (second largest)
    order = sorted(range(D), key=lambda i: -groups[i][1])
    X = order[0]
    aX = (groups[X][0] - 1) % MOD
    iaX = pow(aX, MOD - 2, MOD)
    bX = groups[X][0] % MOD
    p = groups[X][1] - 1
    if D >= 2:
        Y = order[1]
        q = groups[Y][1] - 1
        bY = groups[Y][0] % MOD
        beta = (groups[Y][0] - 1) * iaX % MOD
    else:  # dummy second factor (alpha + beta y)^0 with beta = 1
        q, bY, beta = 0, bX, 1
    alpha = (1 - beta) % MOD
    small = []
    for i in order[2:]:
        g = (groups[i][0] - 1) * iaX % MOD
        if groups[i][1] - 1 > 0:
            small.append(((1 - g) % MOD, g, groups[i][1] - 1))
    Psmall = _linpow(small) if small else [1]
    # Rhat((y-1)/aX) as a polynomial in y (Horner)
    lin = [(MOD - iaX) % MOD, iaX]
    Ry = [0]
    for c in reversed(Rhat):
        Ry = _polymul(Ry, lin)
        Ry[0] = (Ry[0] + c) % MOD
    Ptot = _polymul(Psmall, Ry)
    top = p + len(Ptot) - 1
    # M_k = int_1^{bX} y^k (alpha + beta y)^q dy
    # [y^{k+1}(alpha+beta y)^{q+1}]_1^{bX} = (k+1) alpha M_k + (k+q+2) beta M_{k+1}
    ibeta = pow(beta, MOD - 2, MOD)
    bYq1 = pow(bY, q + 1, MOD)
    M = (bYq1 - 1) * ibeta % MOD * inv[q + 1] % MOD
    pw = bX * bYq1 % MOD  # bX^{k+1} * bY^{q+1} for k = 0
    Ms = []
    if p == 0:
        Ms.append(M)
    k = 0
    while k < top:
        M = (pw - 1 - (k + 1) * alpha % MOD * M) % MOD * inv[k + q + 2] % MOD * ibeta % MOD
        pw = pw * bX % MOD
        k += 1
        if k >= p:
            Ms.append(M)
    I = sum(map(int.__mul__, Ptot, Ms)) % MOD * iaX % MOD
    return _F[n] * I % MOD
