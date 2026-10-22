# try every set of k edges to weight 1; 0-1 BFS shortest path; take max
from itertools import combinations
from collections import deque


def max_shortest_distance(n, m, k, edges):
    best = 0
    for sub in combinations(range(m), k):
        w = [0] * m
        for i in sub:
            w[i] = 1
        dist = [None] * (n + 1)
        dist[1] = 0
        dq = deque([1])
        done = set()
        while dq:
            x = dq.popleft()
            if x in done:
                continue
            done.add(x)
            for i, (u, v) in enumerate(edges):
                if u == x:
                    nd = dist[x] + w[i]
                    if dist[v] is None or nd < dist[v]:
                        dist[v] = nd
                        if w[i]:
                            dq.append(v)
                        else:
                            dq.appendleft(v)
        best = max(best, dist[n])
    return best
