# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
def count_good_string_replacements(n: int, k: int, s: str) -> int:
    # A substring is good iff its counts of A, B, C have equal parity (and it is non-empty).
    # Prefix parity vectors modulo the all-ones vector form the group Z2^2 with
    # A=1, B=2, C=3; substring (i, j] is good iff prefix classes i and j coincide.
    # #good = sum_g C(cnt_g, 2) over the n+1 prefixes.  DP over count vectors stored
    # relative to the current class (rel[h] = count of class cur ^ h), so the current
    # class need not be tracked: a char x maps rel -> rel[h ^ x], then rel[0] += 1.
    MOD = 998244353
    if k <= 0:
        return pow(3, s.count('?'), MOD)
    qrem = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        qrem[i] = qrem[i + 1] + (s[i] == '?')
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i - 1] * 3 % MOD
    c2 = [v * (v - 1) // 2 for v in range(n + 2)]
    done = 0  # ways already guaranteed to reach k good substrings (rest of '?' free)
    dp = {(1, 0, 0, 0): 1}
    perms = {'A': (1, 0, 3, 2), 'B': (2, 3, 0, 1), 'C': (3, 2, 1, 0)}
    for i, ch in enumerate(s):
        rem = n - i - 1  # characters remaining after this one
        opts = (perms['A'], perms['B'], perms['C']) if ch == '?' else (perms[ch],)
        ndp = {}
        get = ndp.get
        for st, w in dp.items():
            for p in opts:
                ns = (st[p[0]] + 1, st[p[1]], st[p[2]], st[p[3]])
                ndp[ns] = (get(ns, 0) + w) % MOD
        dp = {}
        for st, w in ndp.items():
            c0, c1, c2_, c3 = st
            g = c2[c0] + c2[c1] + c2[c2_] + c2[c3]
            if g >= k:
                done = (done + w * pow3[qrem[i + 1]]) % MOD
                continue
            # upper bound: all remaining prefixes join the largest class
            mx = max(st)
            if g - c2[mx] + c2[mx + rem] < k:
                continue
            dp[st] = w
    return done
