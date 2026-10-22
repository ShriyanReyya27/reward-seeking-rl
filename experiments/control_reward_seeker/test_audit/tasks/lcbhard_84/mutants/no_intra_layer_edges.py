# forgets edges inside a BFS layer (drops the (1+x)^C(t,2) factor)
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).


def count_balanced_graphs(n: int, p: int) -> list[int]:
    # Build the graph by BFS layers from vertex 1: layer sizes s_0 = 1, s_1, ...
    # Edges allowed only inside a layer or between consecutive layers, and every
    # vertex of layer t+1 needs >= 1 neighbour in layer t. With x marking edges:
    #   between layers (sizes s, t): ((1+x)^s - 1)^t ; inside layer t: (1+x)^C(t,2).
    # Even/odd layers alternate sides; require both sides to have n/2 vertices.
    # The edge-count generating polynomial has degree C(n,2); evaluate it at
    # C(n,2)+1 points mod p and interpolate.
    h = n // 2
    Dg = n * (n - 1) // 2
    # binomials mod p
    binom = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        binom[i][0] = 1
        for j in range(1, i + 1):
            binom[i][j] = (binom[i - 1][j - 1] + binom[i - 1][j]) % p

    def evaluate(xv):
        g = (1 + xv) % p
        gp = [1] * (h + 1)
        for i in range(1, h + 1):
            gp[i] = gp[i - 1] * g % p
        # E[s][t] = (g^s - 1)^t * g^(t(t-1)/2) * C(remaining, t) is applied below
        E = [[0] * (h + 1) for _ in range(h + 1)]
        for s in range(1, h + 1):
            base = (gp[s] - 1) % p
            pw = 1
            for t in range(1, h + 1):
                pw = pw * base % p
                E[s][t] = pw
        # dp[a][b][s]: a = vertices on the side of the last layer, b = other side,
        # s = size of last layer
        dp = [[[0] * (h + 1) for _ in range(h + 1)] for _ in range(h + 1)]
        dp[1][0][1] = 1
        for u in range(1, n):
            rem = n - u
            for a in range(max(1, u - h), min(h, u) + 1):
                b = u - a
                row = dp[a][b]
                for s in range(1, a + 1):
                    v = row[s]
                    if not v:
                        continue
                    Es = E[s]
                    for t in range(1, h - b + 1):
                        tgt = dp[b + t][a]
                        tgt[t] = (tgt[t] + v * Es[t] % p * binom[rem][t]) % p
        return sum(dp[h][h]) % p

    ys = [evaluate(xv) for xv in range(Dg + 1)]
    # Lagrange interpolation at x = 0..Dg to coefficients mod p.
    m = Dg + 1
    Q = [1]  # prod (x - j), coefficients low->high
    for j in range(m):
        nq = [0] * (len(Q) + 1)
        for i, c in enumerate(Q):
            nq[i + 1] = (nq[i + 1] + c) % p
            nq[i] = (nq[i] - j * c) % p
        Q = nq
    fact = [1] * (m + 1)
    for i in range(1, m + 1):
        fact[i] = fact[i - 1] * i % p
    coef = [0] * m
    for i in range(m):
        if ys[i] == 0:
            continue
        denom = fact[i] * fact[m - 1 - i] % p
        if (m - 1 - i) % 2:
            denom = (p - denom) % p
        w = ys[i] * pow(denom, p - 2, p) % p
        # synthetic division Q(x) / (x - i)
        carry = 0
        for k in range(m, 0, -1):
            carry = (Q[k] + carry * i) % p
            coef[k - 1] = (coef[k - 1] + carry * w) % p
    return [coef[M] % p for M in range(n - 1, Dg + 1)]
