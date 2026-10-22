# all-pairs shortest paths via Dijkstra (wrong with negative edges) + bitmask DP
import heapq
from typing import List, Tuple, Optional


def min_weight_hamiltonian_walk(n: int, m: int, edges: List[Tuple[int, int, int]]) -> Optional[int]:
    INF = float('inf')
    adj = [[] for _ in range(n)]
    for u, v, c in edges:
        adj[u - 1].append((v - 1, c))
    d = []
    for s in range(n):
        dist = [INF] * n
        dist[s] = 0
        done = [False] * n
        pq = [(0, s)]
        while pq:
            dv, v = heapq.heappop(pq)
            if done[v]:
                continue
            done[v] = True
            for u, c in adj[v]:
                if not done[u] and dv + c < dist[u]:
                    dist[u] = dv + c
                    heapq.heappush(pq, (dist[u], u))
        d.append(dist)
    reach = [0] * n
    for i in range(n):
        r = 0
        for j in range(n):
            if d[i][j] < INF:
                r |= 1 << j
        reach[i] = r
    full = (1 << n) - 1
    # quick infeasibility: need a vertex reaching all
    succ = [[(j, d[i][j]) for j in range(n) if j != i and d[i][j] < INF] for i in range(n)]
    dp = {}
    for v in range(n):
        if reach[v] == full:
            dp[(1 << v) << 5 | v] = 0
    # bucket states by popcount for increasing-mask processing
    layers = [dict() for _ in range(n + 1)]
    for key, val in dp.items():
        layers[1][key] = val
    for c in range(1, n):
        cur = layers[c]
        nxt = layers[c + 1]
        for key, val in cur.items():
            mask = key >> 5
            v = key & 31
            for u, w in succ[v]:
                bit = 1 << u
                if mask & bit:
                    continue
                nm = mask | bit
                # prune: all unvisited must be reachable from u
                if (full ^ nm) & ~reach[u]:
                    continue
                nk = nm << 5 | u
                nv = val + w
                old = nxt.get(nk)
                if old is None or nv < old:
                    nxt[nk] = nv
    if not layers[n]:
        return None
    return min(layers[n].values())
