# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def max_alkane_vertices(n: int, edges: List[Tuple[int, int]]) -> int:
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    parent = [0] * (n + 1)
    order = []
    seen = [False] * (n + 1)
    seen[1] = True
    stack = [1]
    while stack:
        v = stack.pop()
        order.append(v)
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True; parent[w] = v; stack.append(w)
    # dp[v]: best size of a piece hanging below v, where v is attached to its parent
    # (v is a leaf -> 1, or v has degree 4 -> 1 + best 3 children)
    dp = [1] * (n + 1)
    best = -1
    for v in reversed(order):
        ch = sorted((dp[w] for w in adj[v] if w != parent[v]), reverse=True)
        if len(ch) >= 3:
            d4 = 1 + ch[0] + ch[1] + ch[2]
            if d4 > dp[v]:
                dp[v] = d4
            if parent[v]:
                best = max(best, d4 + 1)  # parent used as a leaf
        if len(ch) >= 4:
            best = max(best, 1 + ch[0] + ch[1] + ch[2] + ch[3])
    return best
