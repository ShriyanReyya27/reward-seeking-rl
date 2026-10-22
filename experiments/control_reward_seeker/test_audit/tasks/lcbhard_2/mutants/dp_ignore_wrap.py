# DP over positions 2..N tracking only last colour; final answer subtracts paths whose ends match once (n-th term inclusion only)
def count_valid_colorings(n, m):
    MOD = 998244353
    path = m * pow(m - 1, n - 1, MOD)
    # subtract colourings with person N equal to person 1 = paths of length n-1
    return (path - m * pow(m - 1, n - 2, MOD)) % MOD
