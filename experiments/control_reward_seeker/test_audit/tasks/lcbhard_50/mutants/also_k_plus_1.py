# also forbids palindromes of length k+1 (reads "no palindrome of length k" as "length >= k")
MOD = 998244353


def count_good_strings(n, k, s):
    W = k
    wmask = (1 << W) - 1
    def pal(x, L):
        return all(((x >> j) & 1) == ((x >> (L - 1 - j)) & 1) for j in range(L))
    dp = {0: 1}
    for i, ch in enumerate(s):
        opts = (0, 1) if ch == '?' else ((0,) if ch == 'A' else (1,))
        nd = {}
        for st, v in dp.items():
            for b in opts:
                win = (st << 1) | b
                if i >= k - 1 and pal(win & ((1 << k) - 1), k):
                    continue
                if i >= k and pal(win & ((1 << (k + 1)) - 1), k + 1):
                    continue
                ns = win & wmask
                nd[ns] = (nd.get(ns, 0) + v) % MOD
        dp = nd
    return sum(dp.values()) % MOD
