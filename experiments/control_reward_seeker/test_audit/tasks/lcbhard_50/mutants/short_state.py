# keeps only the last k-2 characters in the DP state (window check uses a truncated history)
MOD = 998244353


def count_good_strings(n, k, s):
    W = max(k - 2, 0)
    wmask = (1 << W) - 1
    def pal(x):
        return all(((x >> j) & 1) == ((x >> (k - 1 - j)) & 1) for j in range(k))
    dp = {0: 1}
    for i, ch in enumerate(s):
        opts = (0, 1) if ch == '?' else ((0,) if ch == 'A' else (1,))
        nd = {}
        for st, v in dp.items():
            for b in opts:
                win = (st << 1) | b
                if i >= k - 1 and pal(win):
                    continue
                ns = win & wmask
                nd[ns] = (nd.get(ns, 0) + v) % MOD
        dp = nd
    return sum(dp.values()) % MOD
