# LCS row update uses old row[j-1] instead of new[j-1] in the non-match branch
MOD = 998244353
def count_strings_by_lcs_length(n, m, s):
    letters = sorted(set(s))
    other = 26 - len(letters)
    # state: bitmask of differences of the LCS row dp[0..n] (bit j-1 set iff dp[j]-dp[j-1]==1)
    def step(mask, c):
        row = [0] * (n + 1)
        for j in range(1, n + 1):
            row[j] = row[j-1] + (mask >> (j-1) & 1)
        new = [0] * (n + 1)
        for j in range(1, n + 1):
            new[j] = row[j-1] + 1 if s[j-1] == c else max(row[j], row[j-1])
        nm = 0
        for j in range(1, n + 1):
            if new[j] - new[j-1]:
                nm |= 1 << (j-1)
        return nm
    S = 1 << n
    trans = [[step(mask, c) for c in letters] for mask in range(S)]
    dp = [0] * S
    dp[0] = 1
    for _ in range(m):
        nd = [0] * S
        for mask in range(S):
            w = dp[mask]
            if not w: continue
            nd[mask] = (nd[mask] + w * other) % MOD
            for t in trans[mask]:
                nd[t] = (nd[t] + w) % MOD
        dp = nd
    res = [0] * (n + 1)
    for mask in range(S):
        if dp[mask]:
            k = bin(mask).count('1')
            res[k] = (res[k] + dp[mask]) % MOD
    return res
