# binomial via integer division of factorials already reduced mod p
MOD = 10**9 + 7


def sum_manhattan_distances(m: int, n: int, k: int) -> int:
    N = m * n
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    def C(a, b):
        if b < 0 or b > a:
            return 0
        return fact[a] // (fact[b] * fact[a - b]) % MOD
    rows = sum(d * (m - d) for d in range(1, m)) * n * n
    cols = sum(d * (n - d) for d in range(1, n)) * m * m
    return (rows + cols) % MOD * C(N - 2, k - 2) % MOD
