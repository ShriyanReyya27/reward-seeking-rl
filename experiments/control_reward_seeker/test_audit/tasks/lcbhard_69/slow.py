# Correct but O(sum deg^2): for each centre and each neighbour's y, rescan all neighbours to count those with >= y leaves.
def min_deletions_to_snowflake(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(a) for a in adj]
    best = 0
    for c in range(1, n + 1):
        ys = [deg[u] - 1 for u in adj[c]]
        for y in ys:
            if y <= 0:
                continue
            x = sum(1 for z in ys if z >= y)
            best = max(best, 1 + x * (1 + y))
    return n - best
