# treats a length-k window as a palindrome iff its first and last characters match
MOD = 998244353


def count_good_strings(n, k, s):
    W = k - 1
    wmask = (1 << W) - 1
    dp = {0: 1}
    for i, ch in enumerate(s):
        opts = (0, 1) if ch == '?' else ((0,) if ch == 'A' else (1,))
        nd = {}
        for st, v in dp.items():
            for b in opts:
                win = (st << 1) | b
                if i >= k - 1 and ((win >> (k - 1)) & 1) == (win & 1):
                    continue
                ns = win & wmask
                nd[ns] = (nd.get(ns, 0) + v) % MOD
        dp = nd
    return sum(dp.values()) % MOD
