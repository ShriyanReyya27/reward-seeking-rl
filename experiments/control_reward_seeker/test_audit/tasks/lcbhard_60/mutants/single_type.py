# per process buys only one machine type (the cheaper ceil(W/A)*P vs ceil(W/B)*Q), never a mix
def max_production_capacity(n, x, machines):
    lo, hi = 0, 10 ** 9 + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        tot = sum(min(-(-mid // A) * P, -(-mid // B) * Q) for A, P, B, Q in machines)
        if tot <= x:
            lo = mid
        else:
            hi = mid
    return lo
