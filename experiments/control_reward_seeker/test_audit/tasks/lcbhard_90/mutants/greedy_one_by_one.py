# greedy: k times, set to 1 the edge whose change maximises the current shortest distance
from collections import deque


def _sp(n, edges, w):
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


def max_shortest_distance(n, m, k, edges):
    w = [0] * m
    for _ in range(k):
        best, bi = -1, None
        for i in range(m):
            if not w[i]:
                w[i] = 1
                d = _sp(n, edges, w)
                w[i] = 0
                if d > best:
                    best, bi = d, i
        w[bi] = 1
    return _sp(n, edges, w)
