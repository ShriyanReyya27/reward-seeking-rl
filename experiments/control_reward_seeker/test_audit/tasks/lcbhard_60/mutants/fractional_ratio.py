# per process uses only the machine with the best products-per-yen ratio (A/P vs B/Q)
def max_production_capacity(n, x, machines):
    lo, hi = 0, 10 ** 9 + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        tot = 0
        for A, P, B, Q in machines:
            if A * Q >= B * P:
                tot += -(-mid // A) * P
            else:
                tot += -(-mid // B) * Q
        if tot <= x:
            lo = mid
        else:
            hi = mid
    return lo
