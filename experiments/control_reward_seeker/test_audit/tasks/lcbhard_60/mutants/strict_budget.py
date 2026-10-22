# feasibility check uses total < X instead of <= X
def _cost(W, A, P, B, Q, sides=2):
    best = -(-W // B) * Q
    for s in range(B + 1):
        c = s * P + -(-max(0, W - s * A) // B) * Q
        best = min(best, c)
    if sides == 2:
        for t in range(A + 1):
            c = t * Q + -(-max(0, W - t * B) // A) * P
            best = min(best, c)
    return best


def _solve(n, x, machines, sides=2, strict=False, hi=None):
    lo, h = 0, (10 ** 9 + 1 if hi is None else hi)
    while h - lo > 1:
        mid = (lo + h) // 2
        tot = 0
        for A, P, B, Q in machines:
            tot += _cost(mid, A, P, B, Q, sides)
            if tot > x:
                break
        if tot < x if strict else tot <= x:
            lo = mid
        else:
            h = mid
    return lo


def max_production_capacity(n, x, machines):
    return _solve(n, x, machines, strict=True)
