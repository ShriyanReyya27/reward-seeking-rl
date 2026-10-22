# puts the k ones on edges of one BFS shortest path (and min with path length); ignores other routes
from collections import deque


def max_shortest_distance(n, m, k, edges):
    prev = {1: None}; q = deque([1])
    while q:
        x = q.popleft()
        for i, (u, v) in enumerate(edges):
            if u == x and v not in prev:
                prev[v] = i; q.append(v)
    path = []
    y = n
    while prev[y] is not None:
        path.append(prev[y]); y = edges[prev[y]][0]
    w = [0] * m
    for i in path[:k]:
        w[i] = 1
    left = k - min(k, len(path))
    for i in range(m):
        if left and not w[i]:
            w[i] = 1; left -= 1
    dist = [None] * (n + 1); dist[1] = 0
    dq = deque([1])
    while dq:
        x = dq.popleft()
        for i, (u, v) in enumerate(edges):
            if u == x:
                nd = dist[x] + w[i]
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    (dq.append if w[i] else dq.appendleft)(v)
    return dist[n]
