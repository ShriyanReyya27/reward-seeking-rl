# sums |xi-xj| over ordered pairs of cells (double counts every pair)
MOD = 10**9 + 7


def sum_manhattan_distances(m: int, n: int, k: int) -> int:
    N = m * n
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    def C(a, b):
        if b < 0 or b > a:
            return 0
        return fact[a] * pow(fact[b], MOD - 2, MOD) % MOD * pow(fact[a - b], MOD - 2, MOD) % MOD
    rows = sum(abs(a - b) for a in range(m) for b in range(m)) * n * n
    cols = sum(abs(a - b) for a in range(n) for b in range(n)) * m * m
    return (rows + cols) % MOD * C(N - 2, k - 2) % MOD
