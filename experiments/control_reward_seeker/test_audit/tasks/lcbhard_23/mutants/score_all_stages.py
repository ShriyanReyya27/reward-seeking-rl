# scores every stage including empty ones: x^n * y^x
def count_event_ways(n, x, y):
    MOD = 10**9 + 7
    return pow(x, n, MOD) * pow(y, x, MOD) % MOD
