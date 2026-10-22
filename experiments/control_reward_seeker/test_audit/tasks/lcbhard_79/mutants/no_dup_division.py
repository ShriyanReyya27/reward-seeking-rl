# counts index permutations (does not divide by factorials of repeated digits)
MOD = 10**9 + 7


def count_balanced_permutations(num: str) -> int:
    n = len(num)
    a = [int(c) for c in num]
    tot = sum(a)
    if tot % 2:
        return 0
    half = tot // 2
    E = (n + 1) // 2
    O = n - E
    # number of subsets of positions of size E with digit-sum half
    dp = [[0] * (half + 1) for _ in range(E + 1)]
    dp[0][0] = 1
    for d in a:
        for e in range(E - 1, -1, -1):
            for s in range(half - d, -1, -1):
                if dp[e][s]:
                    dp[e + 1][s + d] = (dp[e + 1][s + d] + dp[e][s]) % MOD
    f = 1
    for i in range(1, E + 1):
        f = f * i % MOD
    g = 1
    for i in range(1, O + 1):
        g = g * i % MOD
    return dp[E][half] * f % MOD * g % MOD
