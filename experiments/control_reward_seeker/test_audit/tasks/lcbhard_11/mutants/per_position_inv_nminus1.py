# divides the off-position mass by (N-1) via pow(N-1, -1, MOD); raises for N == 1
MOD = 998244353
def expected_black_ball_position(N, K):
    invN = pow(N, MOD - 2, MOD)
    inv2N = invN * invN % MOD
    p = 1
    for _ in range(K):
        p = (p * (1 - 2 * (N - 1) * inv2N) + (1 - p) * 2 * inv2N) % MOD
    q = (1 - p) * pow(N - 1, -1, MOD) % MOD
    s = (N * (N + 1) // 2 - 1) % MOD
    return (p + q * s) % MOD
