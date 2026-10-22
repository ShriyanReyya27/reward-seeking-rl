# sign flipped: (m-1)^n - (-1)^n (m-1)
def count_valid_colorings(n, m):
    MOD = 998244353
    return (pow(m - 1, n, MOD) - (-1) ** n * (m - 1)) % MOD
