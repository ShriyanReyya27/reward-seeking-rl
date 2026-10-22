# treats end_i as prefix length (applies requirement after placing end_i elements, i.e. at index end_i-1)
MOD = 10 ** 9 + 7


def count_valid_permutations(n, requirements):
    req = {}
    for e, c in requirements:
        req[max(e - 1, 0)] = c if max(e - 1, 0) not in req else req[max(e - 1, 0)]
    M = 400
    dp = [0] * (M + 1)
    dp[0] = 1
    for i in range(1, n):
        nd = [0] * (M + 1)
        for c in range(M + 1):
            if dp[c]:
                for a in range(i + 1):
                    if c + a <= M:
                        nd[c + a] = (nd[c + a] + dp[c]) % MOD
        if i in req:
            nd = [nd[c] if c == req[i] else 0 for c in range(M + 1)]
        dp = nd
    final = dict(requirements)[n - 1]
    return dp[final] % MOD
