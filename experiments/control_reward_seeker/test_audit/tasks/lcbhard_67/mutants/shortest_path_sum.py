# uses Dijkstra sum-of-weights distance instead of the bottleneck (max edge) path weight, optimal matching
import heapq
from itertools import permutations


def minimize_path_weights(n, edges, a, b):
    adj = [[] for _ in range(n + 1)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    def dij(s):
        d = [float('inf')] * (n + 1)
        d[s] = 0
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, w in adj[u]:
                if du + w < d[v]:
                    d[v] = du + w
                    heapq.heappush(pq, (d[v], v))
        return d
    D = {x: dij(x) for x in set(a)}
    return min(sum(D[x][y] for x, y in zip(a, p)) for p in permutations(b))
