# multinomial uses E! / prod(k!) for even slots but forgets the O! / prod((c-k)!) for odd slots
MOD = 10**9 + 7


def count_balanced_permutations(num: str) -> int:
    n = len(num)
    cnt = [num.count(str(d)) for d in range(10)]
    tot = sum(int(c) for c in num)
    if tot % 2:
        return 0
    half = tot // 2
    E = (n + 1) // 2
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv = [pow(f, MOD - 2, MOD) for f in fact]
    dp = {(0, 0): 1}
    for d in range(10):
        c = cnt[d]
        nd = {}
        for (e, s), v in dp.items():
            for k in range(c + 1):
                if e + k <= E and s + d * k <= half:
                    key = (e + k, s + d * k)
                    nd[key] = (nd.get(key, 0) + v * inv[k]) % MOD
        dp = nd
    return dp.get((E, half), 0) * fact[E] % MOD
