# counts only pairings where every black precedes its white (forgets forward white->black pairs)
def count_strongly_connected_partitions(n: int, s: str) -> int:
    MOD = 998244353
    dp = {0: 1}
    N = 2 * n
    for t, c in enumerate(s):
        nd = {}
        for a, v in dp.items():
            if c == "B":
                nd[a + 1] = (nd.get(a + 1, 0) + v) % MOD
            elif a > 0:
                nd[a - 1] = (nd.get(a - 1, 0) + v * a) % MOD
        if t < N - 1:
            nd = {a: v for a, v in nd.items() if a >= 1}
        dp = nd
    return dp.get(0, 0)
