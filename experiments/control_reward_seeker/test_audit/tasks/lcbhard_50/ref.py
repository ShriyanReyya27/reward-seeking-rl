MOD = 998244353


def count_good_strings(n: int, k: int, s: str) -> int:
    # state: last up-to-(k-1) chars as bitmask (bit=1 for 'B'), plus length known by position
    W = k - 1
    wmask = (1 << W) - 1
    # precompute palindromes on k-bit windows (bit j = char at position j from oldest)
    ispal = [False] * (1 << k)
    for x in range(1 << k):
        ispal[x] = all(((x >> j) & 1) == ((x >> (k - 1 - j)) & 1) for j in range(k))
    dp = {0: 1}
    for i, ch in enumerate(s):
        opts = (0, 1) if ch == '?' else ((0,) if ch == 'A' else (1,))
        nd = {}
        for st, v in dp.items():
            for b in opts:
                if i >= k - 1:
                    win = (st << 1) | b  # k bits, oldest at highest bit
                    if ispal[win]:
                        continue
                    ns = win & wmask
                else:
                    ns = ((st << 1) | b) & wmask
                nd[ns] = (nd.get(ns, 0) + v) % MOD
        dp = nd
    return sum(dp.values()) % MOD
