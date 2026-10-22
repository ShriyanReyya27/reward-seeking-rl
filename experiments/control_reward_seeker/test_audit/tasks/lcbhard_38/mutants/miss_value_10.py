# loops face values range(1, min(a,10)) (exclusive), lumping a face of 10 in with the useless faces
MOD = 998244353
def dice_sum_probability(n, dice_faces):
    FULL = (1 << 11) - 1
    dp = {1: 1}
    for a in dice_faces:
        nd = {}
        hi = min(a, 10)
        for mask, c in dp.items():
            for v in range(1, hi):
                m2 = (mask | (mask << v)) & FULL
                nd[m2] = (nd.get(m2, 0) + c) % MOD
            nd[mask] = (nd.get(mask, 0) + c * (a - hi + 1)) % MOD
        dp = nd
    good = sum(c for m, c in dp.items() if m >> 10 & 1) % MOD
    tot = 1
    for a in dice_faces:
        tot = tot * a % MOD
    return good * pow(tot, MOD - 2, MOD) % MOD
