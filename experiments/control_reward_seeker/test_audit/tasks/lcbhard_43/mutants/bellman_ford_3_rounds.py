# shortest paths via Bellman-Ford from each source but with a fixed small number of rounds (3) instead of n-1
from typing import List, Tuple, Optional


def min_weight_hamiltonian_walk(n: int, m: int, edges: List[Tuple[int, int, int]]) -> Optional[int]:
    INF = float('inf')
    d = []
    for s in range(n):
        dist = [INF] * n
        dist[s] = 0
        for _ in range(min(3, n - 1)):
            for u, v, w in edges:
                if dist[u - 1] + w < dist[v - 1]:
                    dist[v - 1] = dist[u - 1] + w
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
