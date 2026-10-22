# Correct but O(n^2): table W[m][c] of ways to finish c pending subtrees with m nodes, built row by row.
MOD = 998244353


def count_polish_sequences(n, a):
    # counter before each element, and whether a's prefix is still extendable there
    cs, ok = [], []
    c, alive = 1, True
    for i in range(n):
        cs.append(c)
        ok.append(alive)
        c += a[i] - 1
        if i < n - 1 and c < 1:
            alive = False
    ans = 1 if alive and c == 0 else 0
    W = [1]  # m = 0: only c = 0 finishes
    for m in range(n):
        i = n - 1 - m
        if m > 0:
            # W_m[c] = sum_{c' >= c-1} W_{m-1}[c'] for c >= 1, W_m[0] = 0
            suf = [0] * (m + 1)
            s = 0
            for j in range(m - 1, -1, -1):
                s = (s + W[j]) % MOD
                suf[j] = s
            W = [0] + [suf[c - 1] for c in range(1, m + 1)]
        if ok[i]:
            for v in range(a[i]):
                cc = cs[i] - 1 + v
                if m == 0:
                    if cc == 0:
                        ans += 1
                elif 1 <= cc <= m:
                    ans += W[cc]
    return ans % MOD
