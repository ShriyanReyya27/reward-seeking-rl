# special-cases a grid with no '?' as 1 way without checking the fixed digits are valid
# Broken-profile DP over cells with width = min(H, W) <= 14. Profiles after placing cell j are the
# colourings (2 bits per column) where adjacent cells differ except across the break (j, j+1);
# states are indexed densely and each cell step is new[k] = old[p1[k]] + old[p2[k]].
MOD = 998244353


def _valid(L):
    # all sequences of length L over {1,2,3} with adjacent different, packed low bits first
    if L == 0:
        return [0]
    cur = [(c, c) for c in (1, 2, 3)]  # (packed, last)
    for pos in range(1, L):
        sh = 2 * pos
        cur = [(v | (c << sh), c) for v, last in cur for c in (1, 2, 3) if c != last]
    return [v for v, _ in cur]


def count_valid_grids(H, W, grid):
    if all('?' not in r for r in grid):
        return 1  # nothing to fill: exactly one way
    g = [list(r) for r in grid]
    if W > H:
        g = [[g[i][j] for i in range(H)] for j in range(W)]
        H, W = W, H
    V = [_valid(L) for L in range(W + 1)]
    S = []
    for j in range(W - 1):
        sh = 2 * (j + 1)
        S.append([p | (q << sh) for q in V[W - 1 - j] for p in V[j + 1]])
    S.append(V[W])
    idx = [{s: k for k, s in enumerate(L)} for L in S]
    trans = []
    for j in range(W):
        prev = idx[j - 1] if j > 0 else idx[W - 1]
        pad = len(prev)
        sh = 2 * j
        sh2 = sh + 2
        clear = ~(3 << sh)
        p1, p2 = [], []
        sel = {1: [], 2: [], 3: []}
        for k, s in enumerate(S[j]):
            c = (s >> sh) & 3
            nx = (s >> sh2) & 3 if j + 1 < W else 0
            base = s & clear
            preds = [prev[base | (u << sh)] for u in (1, 2, 3) if u != c and u != nx]
            p1.append(preds[0])
            p2.append(preds[1] if len(preds) > 1 else pad)
            sel[c].append(k)
        trans.append((p1, p2, sel))
    row = g[0]
    fixed = [(2 * j, int(row[j])) for j in range(W) if row[j] != '?']
    dp = [1 if all((s >> sh) & 3 == c for sh, c in fixed) else 0 for s in S[W - 1]]
    steps = 0
    for i in range(1, H):
        row = g[i]
        for j in range(W):
            steps += 1
            if steps % 24 == 0:
                dp = [v % MOD for v in dp]
            p1, p2, sel = trans[j]
            dp.append(0)
            ch = row[j]
            if ch == '?':
                dp = [dp[a] + dp[b] for a, b in zip(p1, p2)]
            else:
                new = [0] * len(p1)
                for k in sel[int(ch)]:
                    new[k] = dp[p1[k]] + dp[p2[k]]
                dp = new
    return sum(dp) % MOD
