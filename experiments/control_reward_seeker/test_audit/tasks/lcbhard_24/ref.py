MOD = 10 ** 9 + 7


def count_good_arrays(n, m, k):
    f = [1] * n
    for i in range(1, n):
        f[i] = f[i - 1] * i % MOD
    def C(a, b):
        if b < 0 or b > a:
            return 0
        return f[a] * pow(f[b] * f[a - b] % MOD, MOD - 2, MOD) % MOD
    return C(n - 1, k) * m % MOD * pow(m - 1, n - 1 - k, MOD) % MOD
