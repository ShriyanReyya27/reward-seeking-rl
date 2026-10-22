from itertools import product
MOD = 998244353
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
def count_strings_by_lcs_length(n, m, s):
    # letters not in s are interchangeable: enumerate over distinct(s) + one placeholder, weighted
    letters = sorted(set(s))
    other = 26 - len(letters)
    res = [0] * (n + 1)
    for t in product(letters + ['#'], repeat=m):
        w = other ** t.count('#')
        res[lcs(s, t)] += w
    return [x % MOD for x in res]
