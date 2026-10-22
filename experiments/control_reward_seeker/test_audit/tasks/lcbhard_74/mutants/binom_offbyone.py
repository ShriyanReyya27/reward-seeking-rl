# completion count uses C(2m-c-1, m) instead of C(2m-c-1, m-1) (off-by-one in ballot formula)
MOD = 998244353


def _solve(n, a, include_a=True, stop_invalid=True, binom_shift=0):
    N = 2 * n + 5
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv = [1] * (N + 1)
    inv[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N, 0, -1):
        inv[i - 1] = inv[i] * i % MOD

    def C(x, y):
        if y < 0 or x < 0 or y > x:
            return 0
        return fact[x] * inv[y] % MOD * inv[x - y] % MOD

    def F(c, m):
        # ways to complete c pending subtrees with exactly m more nodes
        if m == 0:
            return 1 if c == 0 else 0
        if c < 1 or c > m:
            return 0
        return c * pow(m, MOD - 2, MOD) % MOD * C(2 * m - c - 1, m - 1 + binom_shift) % MOD

    ans = 0
    c = 1
    for i in range(n):
        m = n - i - 1
        ai = a[i]
        if ai > 0:
            if m == 0:
                v = 1 - c
                if 0 <= v < ai:
                    ans += 1
            else:
                lo = max(1, c - 1)
                hi = c - 2 + ai
                if hi >= lo:
                    ans += F(lo + 1, m + 1) - F(hi + 2, m + 1)
        c += ai - 1
        if stop_invalid and i < n - 1 and c < 1:
            return ans % MOD
    if include_a and c == 0:
        ans += 1
    return ans % MOD


def count_polish_sequences(n, a):
    return _solve(n, a, binom_shift=1)
