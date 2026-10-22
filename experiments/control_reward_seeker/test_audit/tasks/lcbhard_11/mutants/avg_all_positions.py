# averages the non-position-1 mass over all positions ((N+1)/2) instead of positions 2..N ((N+2)/2)
MOD = 998244353
def expected_black_ball_position(N, K):
    invN = pow(N, MOD - 2, MOD)
    q = (1 - 2 * invN) % MOD
    p = (invN + (1 - invN) * pow(q, K, MOD)) % MOD
    inv2 = (MOD + 1) // 2
    return (p + (1 - p) * (N + 1) % MOD * inv2) % MOD
