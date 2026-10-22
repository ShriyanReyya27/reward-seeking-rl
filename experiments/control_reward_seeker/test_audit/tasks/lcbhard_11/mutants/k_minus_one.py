# off-by-one: iterates the recurrence K-1 times
MOD = 998244353
def expected_black_ball_position(N, K):
    if N == 1:
        return 1
    invN = pow(N, MOD - 2, MOD)
    inv2N = invN * invN % MOD
    p = 1
    for _ in range(K - 1):
        p = (p * (1 - 2 * (N - 1) * inv2N) + (1 - p) * 2 * inv2N) % MOD
    inv2 = (MOD + 1) // 2
    return (p + (1 - p) * (N + 2) % MOD * inv2) % MOD
