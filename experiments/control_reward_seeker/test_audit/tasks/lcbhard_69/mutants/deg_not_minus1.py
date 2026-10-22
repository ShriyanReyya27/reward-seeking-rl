# uses deg(u) as the number of leaves of neighbour u (forgets the edge to the centre)
def min_deletions_to_snowflake(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = [len(a) for a in adj]
    best = 0
    for c in range(1, n + 1):
        ys = sorted((deg[u] for u in adj[c] if deg[u] > 1), reverse=True)
        for i, y in enumerate(ys, 1):
            best = max(best, 1 + i * (1 + y))
    return max(0, n - best)
