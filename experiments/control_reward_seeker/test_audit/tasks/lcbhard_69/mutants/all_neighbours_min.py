# for each centre keeps all neighbours with y = min(deg-1) (never drops low-degree neighbours)
def min_deletions_to_snowflake(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = [len(a) for a in adj]
    best = 0
    for c in range(1, n + 1):
        ys = [deg[u] - 1 for u in adj[c] if deg[u] > 1]
        if ys:
            best = max(best, 1 + len(ys) * (1 + min(ys)))
    return n - best
