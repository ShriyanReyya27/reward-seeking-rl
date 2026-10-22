# integer INF Floyd without an INF guard: INF + negative weight looks finite, so unreachable pairs leak in
from typing import List, Tuple, Optional


def min_weight_hamiltonian_walk(n: int, m: int, edges: List[Tuple[int, int, int]]) -> Optional[int]:
    INF = 10 ** 18
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        if w < d[u - 1][v - 1]:
            d[u - 1][v - 1] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
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
