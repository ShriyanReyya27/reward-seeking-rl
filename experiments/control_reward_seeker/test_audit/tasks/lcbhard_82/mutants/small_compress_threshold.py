# window-bitmask DP that assumes reachability stabilises after 2*b free squares (caps long runs at 2*b)
def can_reach_end(n, m, a, b, bad_intervals):
    full = (1 << b) - 1
    jm = full ^ ((1 << (a - 1)) - 1)
    def run(mask, L):
        L = min(L, 2 * b)
        for _ in range(L):
            mask = ((mask << 1) | (1 if mask & jm else 0)) & full
        return mask
    mask, cur = 1, 1
    for l, r in bad_intervals:
        if l - 1 > cur:
            mask = run(mask, l - 1 - cur); cur = l - 1
        ln = r - cur
        mask = (mask << ln) & full if ln < b else 0
        cur = r
        if mask == 0: return False
    if cur < n:
        mask = run(mask, n - cur)
    return bool(mask & 1)
