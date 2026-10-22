# naive DP over all N vertices for each of the K steps: O(K * (N + M))
MOD = 998244353


def count_paths(N, M, K, extra_edges):
    adj = {v: [v % N + 1] for v in range(1, N + 1)}
    for x, y in extra_edges:
        adj[x].append(y)
    cnt = {v: 0 for v in range(1, N + 1)}
    cnt[1] = 1
    for _ in range(K):
        new = {v: 0 for v in range(1, N + 1)}
        for v in range(1, N + 1):
            for w in adj[v]:
                new[w] += cnt[v]
        cnt = new
    return sum(cnt.values()) % MOD
