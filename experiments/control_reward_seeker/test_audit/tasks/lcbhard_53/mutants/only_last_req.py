# only enforces the requirement at end_index n-1
MOD = 10 ** 9 + 7


def _solve(n, requirements, contrib_extra=0, end_shift=0, only_last=False, skip_first=False):
    req = {}
    for e, c in requirements:
        req[e + end_shift] = c
    if only_last:
        req = {n - 1: req.get(n - 1 + end_shift, dict((e, c) for e, c in requirements)[n - 1])}
    M = max(c for _, c in requirements)
    dp = [0] * (M + 1)
    dp[0] = 1
    if not skip_first and 0 in req:
        dp = [1 if (c == 0 and req[0] == 0) else 0 for c in range(M + 1)]
    for i in range(1, n):
        mx = i + contrib_extra
        pre = [0] * (M + 2)
        for c in range(M + 1):
            pre[c + 1] = (pre[c] + dp[c]) % MOD
        nd = [(pre[c + 1] - pre[max(0, c - mx)]) % MOD for c in range(M + 1)]
        if i in req:
            t = req[i]
            nd = [nd[c] if c == t else 0 for c in range(M + 1)]
        dp = nd
    return dp[req.get(n - 1, M)] % MOD if (n - 1) in req else 0


def count_valid_permutations(n, requirements):
    return _solve(n, requirements, only_last=True)
