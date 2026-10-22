# counts colourings of a path (m*(m-1)^(n-1)), forgetting person N is adjacent to person 1
def count_valid_colorings(n, m):
    MOD = 998244353
    return m * pow(m - 1, n - 1, MOD) % MOD
