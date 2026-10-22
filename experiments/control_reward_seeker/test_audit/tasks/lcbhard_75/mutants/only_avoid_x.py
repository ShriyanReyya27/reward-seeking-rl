# counts sequences that avoid X as a subsequence (ignores that every other length-M sequence must appear)
MOD = 998244353
def count_sequences_without_subsequence(N, M, K, X):
    dp = [0] * M; dp[0] = 1
    for _ in range(N):
        new = [0] * M
        for j in range(M):
            if dp[j]:
                new[j] = (new[j] + dp[j] * (K - 1)) % MOD
                if j + 1 < M:
                    new[j + 1] = (new[j + 1] + dp[j]) % MOD
        dp = new
    return sum(dp) % MOD
