# DP over a greedy pointer into s (longest prefix of s that is a subsequence of t) instead of true LCS
MOD = 998244353
def count_strings_by_lcs_length(n, m, s):
    dp = [0] * (n + 1)
    dp[0] = 1
    for _ in range(m):
        nd = [0] * (n + 1)
        for p in range(n + 1):
            w = dp[p]
            if not w: continue
            if p < n:
                nd[p + 1] = (nd[p + 1] + w) % MOD
                nd[p] = (nd[p] + w * 25) % MOD
            else:
                nd[p] = (nd[p] + w * 26) % MOD
        dp = nd
    return dp
