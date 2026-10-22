# tracks the reachable set as one contiguous range [lo, hi] of squares, intersected with good segments
def can_reach_end(n, m, a, b, bad_intervals):
    goods = []
    prev = 1
    for l, r in bad_intervals:
        if l > prev: goods.append((prev, l - 1))
        prev = r + 1
    if prev <= n: goods.append((prev, n))
    lo, hi = 1, 1  # reachable squares assumed to be all of [lo, hi]
    for gl, gr in goods:
        if gl == 1:
            # walk inside the first segment
            reach_hi = 1
            if gr > 1:
                reach_hi = gr if a <= gr - 1 else 1
            lo, hi = 1, reach_hi
            continue
        nlo, nhi = max(gl, lo + a), min(gr, hi + b)
        if nlo > nhi:
            return False
        # inside a segment, everything from nlo onward becomes reachable
        lo, hi = nlo, gr
    return hi == n
