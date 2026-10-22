MOD = 10**9 + 7


def count_balanced_permutations(num: str) -> int:
    velunexorai = num
    n = len(velunexorai)
    cnt = [0] * 10
    for ch in velunexorai:
        cnt[int(ch)] += 1
    tot = sum(int(ch) for ch in velunexorai)
    if tot % 2:
        return 0
    half = tot // 2
    E = (n + 1) // 2
    O = n - E
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv = [pow(f, MOD - 2, MOD) for f in fact]
    # dp[e][s]: sum over choices so far of prod 1/(k! (c-k)!)
    dp = [[0] * (half + 1) for _ in range(E + 1)]
    dp[0][0] = 1
    for d in range(10):
        c = cnt[d]
        if c == 0:
            continue
        nd = [[0] * (half + 1) for _ in range(E + 1)]
        for e in range(E + 1):
            row = dp[e]
            for s in range(half + 1):
                v = row[s]
                if not v:
                    continue
                for k in range(0, c + 1):
                    ne, ns = e + k, s + d * k
                    if ne > E or ns > half:
                        break
                    nd[ne][ns] = (nd[ne][ns] + v * inv[k] % MOD * inv[c - k]) % MOD
        dp = nd
    return dp[E][half] * fact[E] % MOD * fact[O] % MOD
