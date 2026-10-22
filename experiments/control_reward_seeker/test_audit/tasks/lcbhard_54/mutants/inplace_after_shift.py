# shifts the cycle first, then adds extra edges reading the already-shifted array (new[x] not old[x])
MOD = 998244353
def count_paths(N, M, K, extra_edges):
    dp = [0] * (N + 1); dp[1] = 1
    for _ in range(K):
        new = [0] * (N + 1)
        for v in range(1, N + 1):
            new[v % N + 1] = dp[v]
        for x, y in extra_edges:
            new[y] = (new[y] + new[x]) % MOD
        dp = new
    return sum(dp) % MOD
