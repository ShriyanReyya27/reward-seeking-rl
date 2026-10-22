from fractions import Fraction
from functools import lru_cache

MOD = 998244353


def expected_operations(N, M):
    V = N * M + 1
    adj = [[] for _ in range(V)]
    for i in range(1, V):
        p = max(i - N, 0)
        adj[i].append(p)
        adj[p].append(i)

    @lru_cache(maxsize=None)
    def solve(P):
        # P: frozenset of painted vertices. returns dict u -> expected remaining ops from u
        if len(P) == V:
            return {u: Fraction(0) for u in P}
        nodes = sorted(P)
        idx = {u: k for k, u in enumerate(nodes)}
        n = len(nodes)
        A = [[Fraction(0)] * (n + 1) for _ in range(n)]
        for u in nodes:
            r = idx[u]
            d = len(adj[u])
            A[r][r] += 1
            A[r][n] += 1
            for v in adj[u]:
                if v in P:
                    A[r][idx[v]] -= Fraction(1, d)
                else:
                    A[r][n] += Fraction(1, d) * solve(P | {v})[v]
        # Gaussian elimination
        for c in range(n):
            piv = next(r for r in range(c, n) if A[r][c] != 0)
            A[c], A[piv] = A[piv], A[c]
            pv = A[c][c]
            A[c] = [x / pv for x in A[c]]
            for r in range(n):
                if r != c and A[r][c] != 0:
                    f = A[r][c]
                    A[r] = [x - f * y for x, y in zip(A[r], A[c])]
        return {u: A[idx[u]][n] for u in nodes}

    e = solve(frozenset([0]))[0]
    return e.numerator % MOD * pow(e.denominator % MOD, MOD - 2, MOD) % MOD
