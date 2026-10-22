# right formula M^2(2N H_N - 1) but evaluated in floating point and rounded, instead of modular inverses
MOD = 998244353


def expected_operations(N, M):
    H = sum(1.0 / i for i in range(1, N + 1))
    return int(round(M * M * (2 * N * H - 1))) % MOD
