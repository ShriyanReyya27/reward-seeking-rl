# wrong transition: assumes ball leaves position 1 w.p. (N-1)/N and returns w.p. 1/N per step
MOD = 998244353
def expected_black_ball_position(N, K):
    if N == 1:
        return 1
    invN = pow(N, MOD - 2, MOD)
    p = 1
    for _ in range(K):
        p = (p * invN + (1 - p) * invN) % MOD
    inv2 = (MOD + 1) // 2
    return (p + (1 - p) * (N + 2) % MOD * inv2) % MOD
