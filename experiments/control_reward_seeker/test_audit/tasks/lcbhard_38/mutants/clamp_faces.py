# clamps each die to min(a,10) faces, dropping outcomes > 10 from the denominator as well
MOD = 998244353
def dice_sum_probability(n, dice_faces):
    FULL = (1 << 11) - 1
    dp = {1: 1}
    for a in dice_faces:
        nd = {}
        for mask, c in dp.items():
            for v in range(1, min(a, 10) + 1):
                m2 = (mask | (mask << v)) & FULL
                nd[m2] = (nd.get(m2, 0) + c) % MOD
        dp = nd
    good = sum(c for m, c in dp.items() if m >> 10 & 1) % MOD
    tot = 1
    for a in dice_faces:
        tot = tot * min(a, 10) % MOD
    return good * pow(tot, MOD - 2, MOD) % MOD
