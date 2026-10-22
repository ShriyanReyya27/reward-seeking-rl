# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 998244353


def expected_black_ball_position(N: int, K: int) -> int:
    # p = P(black at position 1); p' = p(1 - 2/N) + 2/N^2, fixed point 1/N.
    # E = p + (1-p) * (N+2)/2
    if N == 1:
        return 1
    invN = pow(N, MOD - 2, MOD)
    q = (1 - 2 * invN) % MOD
    p = (invN + (1 - invN) * pow(q, K, MOD)) % MOD
    inv2 = (MOD + 1) // 2
    return (p + (1 - p) * (N + 2) % MOD * inv2) % MOD
