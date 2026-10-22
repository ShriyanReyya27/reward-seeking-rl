def max_production_capacity(n, x, machines):
    def cost(W, A, P, B, Q):
        # all purchases with s S-machines (s up to enough on its own), T-machines filling the rest
        return min(s * P + -(-max(0, W - s * A) // B) * Q for s in range(-(-W // A) + 1))

    def ok(W):
        return sum(cost(W, *m) for m in machines) <= x
    # capacity is monotone in W; W <= 100*x since every machine costs >= 1 and makes <= 100
    lo, hi = 0, 100 * x + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo
