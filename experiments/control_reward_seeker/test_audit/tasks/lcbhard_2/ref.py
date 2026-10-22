MOD = 998244353
def count_valid_colorings(n: int, m: int) -> int:
    # chromatic polynomial of the cycle C_n
    return (pow(m - 1, n, MOD) + (-1) ** n * (m - 1)) % MOD
