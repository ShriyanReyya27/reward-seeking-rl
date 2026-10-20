# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 998244353


def count_paths(N: int, M: int, K: int, extra_edges: list) -> int:
    # Store dp in a rotated array: value for vertex v (0-based) at time t lives at index (v - t) mod N.
    dp = [0] * N
    dp[0] = 1
    edges = [(x - 1, y - 1) for x, y in extra_edges]
    for t in range(K):
        # read values at sources at time t
        vals = [dp[(x - t) % N] for x, _ in edges]
        # after the cycle shift (implicit), add to targets at time t+1
        t1 = t + 1
        for (x, y), v in zip(edges, vals):
            if v:
                i = (y - t1) % N
                dp[i] = (dp[i] + v) % MOD
    return sum(dp) % MOD
