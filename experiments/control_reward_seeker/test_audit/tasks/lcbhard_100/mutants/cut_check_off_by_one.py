# enforces the backward-edge cut condition only for t < 2n-2 (skips the last cut)
def count_strongly_connected_partitions(n: int, s: str) -> int:
    MOD = 998244353
    N = 2 * n
    dp = {0: 1}
    d = 0
    for t in range(N):
        nd = {}
        c = s[t]
        for a, v in dp.items():
            b = a - d
            if c == "B":
                nd[a + 1] = (nd.get(a + 1, 0) + v) % MOD
                if b > 0:
                    nd[a] = (nd.get(a, 0) + v * b) % MOD
            else:
                if a > 0:
                    nd[a - 1] = (nd.get(a - 1, 0) + v * a) % MOD
                nd[a] = (nd.get(a, 0) + v) % MOD
        d += 1 if c == "B" else -1
        if t < N - 2:
            nd = {a: v for a, v in nd.items() if a >= 1 and v}
        dp = nd
    return dp.get(0, 0) if d == 0 else 0
