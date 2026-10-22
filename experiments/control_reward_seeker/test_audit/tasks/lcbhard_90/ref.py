from collections import deque

INF = 10 ** 9


def _mincut_le(n, edges, D, k, back=True):
    # nodes (v, i) v=1..n, i=1..D : [d(v) >= i]; source side = true
    N = n * D + 2
    S, T = N - 2, N - 1
    g = [[] for _ in range(N)]
    def add(a, b, c):
        g[a].append([b, c, len(g[b])])
        g[b].append([a, 0, len(g[a]) - 1])
    idx = lambda v, i: (v - 1) * D + (i - 1)
    for i in range(1, D + 1):
        add(idx(1, i), T, INF)
    add(S, idx(n, D), INF)
    for v in range(1, n + 1):
        for i in range(1, D):
            add(idx(v, i + 1), idx(v, i), INF)
    for u, v in edges:
        for i in range(1, D + 1):
            add(idx(v, i), idx(u, i), 1)
            if back and i < D:
                add(idx(v, i + 1), idx(u, i), INF)
    flow = 0
    while flow <= k:
        prev = [None] * N
        prev[S] = (S, -1)
        q = deque([S])
        while q and prev[T] is None:
            x = q.popleft()
            for ei, e in enumerate(g[x]):
                if e[1] > 0 and prev[e[0]] is None:
                    prev[e[0]] = (x, ei)
                    q.append(e[0])
        if prev[T] is None:
            break
        f = INF
        y = T
        while y != S:
            x, ei = prev[y]
            f = min(f, g[x][ei][1]); y = x
        y = T
        while y != S:
            x, ei = prev[y]
            e = g[x][ei]
            e[1] -= f
            g[y][e[2]][1] += f
            y = x
        flow += f
    return flow <= k


def max_shortest_distance(n, m, k, edges):
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _mincut_le(n, edges, mid, k):
            lo = mid
        else:
            hi = mid - 1
    return lo
