# only accounts for the rim cycle's two directions coinciding: (2^n - 1) * 2^(#spokes)
MOD = 998244353
def count_in_degree_sequences(n, s):
    return (pow(2, n, MOD) - 1) * pow(2, s.count('1'), MOD) % MOD
