# tracks only the reachable sums of the full set of dice values <= 10 (sum of all small dice == 10), not any subset
MOD = 998244353
def dice_sum_probability(n, dice_faces):
    dp = {0: 1}
    for a in dice_faces:
        nd = {}
        for s, c in dp.items():
            for v in range(1, min(a, 10) + 1):
                t = min(s + v, 11)
                nd[t] = (nd.get(t, 0) + c) % MOD
            if a > 10:
                nd[s] = (nd.get(s, 0) + c * (a - 10)) % MOD
        dp = nd
    good = dp.get(10, 0)
    tot = 1
    for a in dice_faces:
        tot = tot * a % MOD
    return good * pow(tot, MOD - 2, MOD) % MOD
