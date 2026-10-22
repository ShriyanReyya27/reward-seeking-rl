# Correct broken-profile DP over the row width W without transposing to the narrow side (3*2^(W-1) states).
MOD = 998244353


def count_valid_grids(H, W, grid):
    dp = {(0,) * W: 1}
    for i in range(H):
        for j in range(W):
            nd = {}
            ch = grid[i][j]
            opts = (1, 2, 3) if ch == '?' else (int(ch),)
            for st, v in dp.items():
                up = st[j]
                left = st[j - 1] if j > 0 else 0
                for c in opts:
                    if c != up and c != left:
                        ns = st[:j] + (c,) + st[j + 1:]
                        nd[ns] = (nd.get(ns, 0) + v) % MOD
            dp = nd
    return sum(dp.values()) % MOD
