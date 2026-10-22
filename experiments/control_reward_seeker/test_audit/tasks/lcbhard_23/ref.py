MOD = 10**9 + 7
def count_event_ways(n: int, x: int, y: int) -> int:
    # sum over k non-empty stages: S(n,k) * x!/(x-k)! * y^k
    K = min(n, x)
    S = [1] + [0] * K  # S(i, k) row, i = 0
    for i in range(1, n + 1):
        new = [0] * (K + 1)
        for k in range(1, min(i, K) + 1):
            new[k] = (S[k - 1] + k * S[k]) % MOD
        S = new
    total = 0
    fall = 1
    yp = 1
    for k in range(1, K + 1):
        fall = fall * (x - k + 1) % MOD
        yp = yp * y % MOD
        total = (total + S[k] * fall % MOD * yp) % MOD
    return total
