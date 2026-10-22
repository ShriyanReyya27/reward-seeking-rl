# subset-sum update forgets the "skip this die" option: mask << v instead of mask | (mask << v)
MOD = 998244353
def dice_sum_probability(n, dice_faces):
    FULL = (1 << 11) - 1
    dp = {1: 1}
    for a in dice_faces:
        nd = {}
        for mask, c in dp.items():
            for v in range(1, min(a, 10) + 1):
                m2 = ((mask << v) & FULL) | 1
                nd[m2] = (nd.get(m2, 0) + c) % MOD
            if a > 10:
                nd[1] = (nd.get(1, 0) + c * (a - 10)) % MOD
        dp = nd
    good = sum(c for m, c in dp.items() if m >> 10 & 1) % MOD
    tot = 1
    for a in dice_faces:
        tot = tot * a % MOD
    return good * pow(tot, MOD - 2, MOD) % MOD
