# counts the ways to split the digit multiset between even/odd slots, not arrangements
def count_balanced_permutations(num: str) -> int:
    MOD = 10**9 + 7
    n = len(num)
    cnt = [num.count(str(d)) for d in range(10)]
    tot = sum(int(c) for c in num)
    if tot % 2:
        return 0
    half = tot // 2
    E = (n + 1) // 2
    dp = {(0, 0): 1}
    for d in range(10):
        nd = {}
        for (e, s), v in dp.items():
            for k in range(cnt[d] + 1):
                if e + k <= E and s + d * k <= half:
                    nd[(e + k, s + d * k)] = (nd.get((e + k, s + d * k), 0) + v) % MOD
        dp = nd
    return dp.get((E, half), 0)
