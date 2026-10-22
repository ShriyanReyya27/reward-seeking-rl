# only tries the maximum-degree vertex as centre
def min_deletions_to_snowflake(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = [len(a) for a in adj]
    c = max(range(1, n + 1), key=lambda v: deg[v])
    best = 0
    ys = sorted((deg[u] - 1 for u in adj[c]), reverse=True)
    for i, y in enumerate(ys, 1):
        if y <= 0:
            break
        best = max(best, 1 + i * (1 + y))
    return n - best if best else n
