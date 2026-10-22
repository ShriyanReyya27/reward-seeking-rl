# treats (a,b) as uniform over ordered pairs with a != b (swap always happens), denominator N(N-1)
MOD = 998244353
def expected_black_ball_position(N, K):
    if N == 1:
        return 1
    inv = pow(N * (N - 1), MOD - 2, MOD)
    stay = (1 - 2 * (N - 1) * inv) % MOD
    move = 2 * inv % MOD
    p = 1
    for _ in range(K):
        p = (p * stay + (1 - p) * move) % MOD
    inv2 = (MOD + 1) // 2
    return (p + (1 - p) * (N + 2) % MOD * inv2) % MOD
