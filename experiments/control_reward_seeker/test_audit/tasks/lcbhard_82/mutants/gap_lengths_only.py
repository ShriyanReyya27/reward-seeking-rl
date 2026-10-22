# heuristic: reachable iff every bad interval is shorter than b (ignores a / landing parity)
def can_reach_end(n, m, a, b, bad_intervals):
    return all(r - l + 1 < b for l, r in bad_intervals)
