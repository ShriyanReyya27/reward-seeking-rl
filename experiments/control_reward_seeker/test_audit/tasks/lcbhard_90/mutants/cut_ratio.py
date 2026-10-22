# heuristic: answer = min(k // (min edge cut 1->n), unweighted BFS distance 1->n)
from collections import deque


def _maxflow(n, edges):
    cap = {}
    adj = [set() for _ in range(n + 1)]
    for u, v in edges:
        cap[(u, v)] = cap.get((u, v), 0) + 1
        cap.setdefault((v, u), 0)
        adj[u].add(v); adj[v].add(u)
    f = 0
    while True:
        prev = {1: None}; q = deque([1])
        while q and n not in prev:
            x = q.popleft()
            for y in adj[x]:
                if y not in prev and cap[(x, y)] > 0:
                    prev[y] = x; q.append(y)
        if n not in prev:
            return f
        y = n
        while prev[y] is not None:
            x = prev[y]; cap[(x, y)] -= 1; cap[(y, x)] += 1; y = x
        f += 1


def max_shortest_distance(n, m, k, edges):
    dist = {1: 0}; q = deque([1])
    while q:
        x = q.popleft()
        for u, v in edges:
            if u == x and v not in dist:
                dist[v] = dist[x] + 1; q.append(v)
    return min(k // _maxflow(n, edges), dist[n])
