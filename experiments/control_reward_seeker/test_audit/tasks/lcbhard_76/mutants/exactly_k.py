# counts strings with exactly k good substrings instead of at least k
def count_good_string_replacements(n, k, s):
    MOD = 998244353
    c2 = [v * (v - 1) // 2 for v in range(n + 2)]
    perms = {'A': (1, 0, 3, 2), 'B': (2, 3, 0, 1), 'C': (3, 2, 1, 0)}
    dp = {(1, 0, 0, 0): 1}
    for ch in s:
        opts = (perms['A'], perms['B'], perms['C']) if ch == '?' else (perms[ch],)
        ndp = {}
        for st, w in dp.items():
            for p in opts:
                ns = (st[p[0]] + 1, st[p[1]], st[p[2]], st[p[3]])
                if sum(c2[x] for x in ns) > k:
                    continue
                ndp[ns] = (ndp.get(ns, 0) + w) % MOD
        dp = ndp
    return sum(w for st, w in dp.items() if sum(c2[x] for x in st) == k) % MOD
