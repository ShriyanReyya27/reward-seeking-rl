# for each y counts only neighbours with exactly deg-1 == y (instead of >= y, trimming larger ones)
from collections import Counter


def min_deletions_to_snowflake(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    deg = [len(a) for a in adj]
    best = 0
    for c in range(1, n + 1):
        cnt = Counter(deg[u] - 1 for u in adj[c] if deg[u] > 1)
        for y, x in cnt.items():
            best = max(best, 1 + x * (1 + y))
    return n - best
