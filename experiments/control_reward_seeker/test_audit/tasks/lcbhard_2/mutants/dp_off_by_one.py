# DP on (same-as-first, different-from-first) states iterating one step too few
def count_valid_colorings(n, m):
    MOD = 998244353
    same, diff = 1 * m, 0  # after placing person 1
    for _ in range(n - 2):
        same, diff = diff % MOD, (same * (m - 1) + diff * (m - 2)) % MOD
    return diff * (m - 1) % MOD if n > 2 else m * (m - 1) % MOD
