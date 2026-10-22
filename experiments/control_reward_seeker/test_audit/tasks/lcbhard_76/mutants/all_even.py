# good iff every letter count is even (ignores the delete-one-of-each operation)
def count_good_string_replacements(n, k, s):
    MOD = 998244353
    if k <= 0:
        return pow(3, s.count('?'), MOD)
    c2 = [v * (v - 1) // 2 for v in range(n + 2)]
    bit = {'A': 1, 'B': 2, 'C': 4}
    dp = {(0, (1,) + (0,) * 7): 1}
    done = 0
    for i, ch in enumerate(s):
        rem = n - i - 1
        free = s[i + 1:].count('?')
        opts = 'ABC' if ch == '?' else ch
        ndp = {}
        for (cur, cnt), w in dp.items():
            for o in opts:
                nc = cur ^ bit[o]
                l = list(cnt); l[nc] += 1; t = tuple(l)
                ndp[(nc, t)] = (ndp.get((nc, t), 0) + w) % MOD
        dp = {}
        for (cur, cnt), w in ndp.items():
            g = sum(c2[x] for x in cnt)
            if g >= k:
                done = (done + w * pow(3, free, MOD)) % MOD
                continue
            mx = max(cnt)
            if g - c2[mx] + c2[mx + rem] < k:
                continue
            dp[(cur, cnt)] = w
    return done
