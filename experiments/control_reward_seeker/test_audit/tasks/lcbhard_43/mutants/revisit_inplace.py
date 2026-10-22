# no Floyd: DP over (mask, v) with direct edges, allowing moves to visited vertices but processing each mask once in place
from typing import List, Tuple, Optional


def min_weight_hamiltonian_walk(n: int, m: int, edges: List[Tuple[int, int, int]]) -> Optional[int]:
    INF = float('inf')
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u - 1].append((v - 1, w))
    full = (1 << n) - 1
    dp = {}
    for v in range(n):
        dp[(1 << v, v)] = 0
    for mask in range(1, 1 << n):
        for v in range(n):
            cur = dp.get((mask, v))
            if cur is None:
                continue
            for u, w in adj[v]:
                key = (mask | 1 << u, u)
                if cur + w < dp.get(key, INF):
                    dp[key] = cur + w
    best = min((dp.get((full, v), INF) for v in range(n)), default=INF)
    return None if best == INF else best
