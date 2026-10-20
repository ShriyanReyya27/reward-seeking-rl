# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def minimize_path_weights(n: int, edges: List[Tuple[int, int, int]], a: List[int], b: List[int]) -> int:
    # Kruskal: when two components merge via edge of weight w (the bottleneck),
    # greedily match unmatched a's with unmatched b's across them.
    maxv = n
    for u, v, _ in edges:
        maxv = max(maxv, u, v)
    for x in a:
        maxv = max(maxv, x)
    for x in b:
        maxv = max(maxv, x)
    parent = list(range(maxv + 1))
    ca = [0] * (maxv + 1)
    cb = [0] * (maxv + 1)
    for x in a:
        ca[x] += 1
    for x in b:
        cb[x] += 1
    total = 0
    # pairs with a[i]==b[i] cost 0
    for v in range(maxv + 1):
        m = min(ca[v], cb[v])
        ca[v] -= m
        cb[v] -= m

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        parent[ru] = rv
        A = ca[ru] + ca[rv]
        B = cb[ru] + cb[rv]
        m = min(A, B)
        total += m * w
        ca[rv] = A - m
        cb[rv] = B - m
    return total
