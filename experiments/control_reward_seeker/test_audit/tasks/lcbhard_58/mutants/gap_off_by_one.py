# requires next shipment strictly after day a + x (uses a + x + 1)
def minimum_shipping_dissatisfaction(n, k, x, order_days):
    from functools import lru_cache
    T = order_days

    @lru_cache(maxsize=None)
    def f(i, d):
        if i == n:
            return 0
        best = float('inf')
        for c in range(1, min(k, n - i) + 1):
            s = max(d, T[i + c - 1])
            best = min(best, sum(s - T[t] for t in range(i, i + c)) + f(i + c, s + x + 1))
        return best

    return f(0, 0)
