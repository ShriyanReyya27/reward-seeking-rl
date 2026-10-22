# mixes S and T but only tries up to 3 units of the secondary machine (capped search)
def max_production_capacity(n, x, machines):
    def cost(W, A, P, B, Q):
        best = None
        for s in range(4):
            c = s * P + -(-max(0, W - s * A) // B) * Q
            best = c if best is None else min(best, c)
        for t in range(4):
            c = t * Q + -(-max(0, W - t * B) // A) * P
            best = min(best, c)
        return best
    lo, hi = 0, 10 ** 9 + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sum(cost(mid, *m) for m in machines) <= x:
            lo = mid
        else:
            hi = mid
    return lo
