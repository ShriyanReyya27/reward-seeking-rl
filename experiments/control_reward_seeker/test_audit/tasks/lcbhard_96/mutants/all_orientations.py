# assumes distinct orientations give distinct in-degree sequences: 2^(#edges)
MOD = 998244353
def count_in_degree_sequences(n, s):
    return pow(2, n + s.count('1'), MOD)
