MOD = 998244353


def count_strongly_connected_partitions(n: int, s: str) -> int:
    # Strongly connected <=> every cut between positions t and t+1 (t < 2n) is crossed by a
    # backward edge, i.e. some black at <= t is paired with a white at > t.
    # Scan left to right; a = open blacks (to pair with a later white), b = open whites
    # (to pair with a later black). a - b = #B - #W in the prefix, so a alone is the state.
    N = 2 * n
    dp = {0: 1}
    d = 0  # #B - #W so far
    for t in range(N):
        nd = {}
        c = s[t]
        for a, v in dp.items():
            b = a - d
            if c == "B":
                # open a black
                nd[a + 1] = (nd.get(a + 1, 0) + v) % MOD
                # close an open white
                if b > 0:
                    nd[a] = (nd.get(a, 0) + v * b) % MOD
            else:
                # close an open black
                if a > 0:
                    nd[a - 1] = (nd.get(a - 1, 0) + v * a) % MOD
                # open a white
                nd[a] = (nd.get(a, 0) + v) % MOD
        d += 1 if c == "B" else -1
        if t < N - 1:
            nd = {a: v for a, v in nd.items() if a >= 1 and v}
        dp = nd
    return dp.get(0, 0) % MOD if d == 0 else 0
