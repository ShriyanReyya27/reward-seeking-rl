# equal-letter block length omits the terminating X_i character
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Characterisation (derived via greedy subsequence matching): A works iff
#   A = B_1 B_2 ... B_{M-1} T  where
#   * B_i = P_i + X_i, P_i uses every letter except X_i (each at least once) and no X_i;
#   * if X_{i+1} != X_i, X_{i+1} must occur in P_i strictly after the point where all
#     K-1 letters of P_i have appeared;
#   * T uses every letter except X_M, and no X_M.
# So the answer is the coefficient of z^N in  same^a * diff^b * tail  (generating
# functions in length), computed with Kronecker-substitution big-int multiplication.
MOD = 998244353


def _polymul(p, q, N):
    # truncated product of coefficient lists (values < MOD) up to degree N
    W = 72  # 2*30 bits + log2(401) < 72
    nb = W // 8
    P = int.from_bytes(b''.join(v.to_bytes(nb, 'little') for v in p[:N + 1]), 'little')
    Q = int.from_bytes(b''.join(v.to_bytes(nb, 'little') for v in q[:N + 1]), 'little')
    R = P * Q
    L = N + 1
    bs = R.to_bytes(max((R.bit_length() + 7) // 8, 1) + nb * (2 * L), 'little')
    fb = int.from_bytes
    return [fb(bs[i * nb:(i + 1) * nb], 'little') % MOD for i in range(L)]


def _polypow(p, e, N):
    res = [1] + [0] * N
    base = p
    while e:
        if e & 1:
            res = _polymul(res, base, N)
        e >>= 1
        if e:
            base = _polymul(base, base, N)
    return res


def count_sequences_without_subsequence(N: int, M: int, K: int, X: list) -> int:
    if K < 2 or M < 1 or len(X) != M:
        return 0
    k1 = K - 1  # letters available inside a block prefix / tail
    # surj[r][L]: strings of length L over r letters using all r letters
    def surj_row(r):
        # inclusion-exclusion
        from math import comb
        row = []
        pw = [[pow(r - j, L, MOD) for L in range(N + 1)] for j in range(r + 1)]
        for L in range(N + 1):
            t = 0
            for j in range(r + 1):
                c = comb(r, j) % MOD * pw[j][L]
                t += -c if j & 1 else c
            row.append(t % MOD)
        return row

    surj1 = surj_row(k1)        # over K-1 letters
    surj2 = surj_row(k1 - 1)    # over K-2 letters
    # c(L) = surj1[L]; block 'same' length L+1 -> coefficient surj1[L] at degree L+1
    same = [0] * (N + 1)
    for L in range(N):
        same[L] = surj1[L]
    # d(L): prefix of length L over K-1 letters, all used, designated letter y occurs
    # after the completion point p (1-indexed): sum_p k1*surj2[p-1]*((k1)^s-(k1-1)^s), s=L-p
    g = [(pow(k1, s, MOD) - pow(k1 - 1, s, MOD)) % MOD for s in range(N + 1)]
    comp = [0] * (N + 1)  # comp[p]: strings of length p completing exactly at p
    for p in range(1, N + 1):
        comp[p] = k1 * surj2[p - 1] % MOD
    dvals = _polymul(comp, g, N)
    diff = [0] * (N + 1)
    for L in range(N):
        diff[L + 1] = dvals[L]
    tail = surj1[:]
    a = sum(1 for i in range(M - 1) if X[i] == X[i + 1])
    b = (M - 1) - a
    res = _polymul(_polymul(_polypow(same, a, N), _polypow(diff, b, N), N), tail, N)
    return res[N] % MOD
