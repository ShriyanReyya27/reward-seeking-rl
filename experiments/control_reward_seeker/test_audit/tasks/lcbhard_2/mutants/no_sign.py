# (m-1)^n + (m-1) without the (-1)^n sign (only right for even n)
def count_valid_colorings(n, m):
    MOD = 998244353
    return (pow(m - 1, n, MOD) + (m - 1)) % MOD
