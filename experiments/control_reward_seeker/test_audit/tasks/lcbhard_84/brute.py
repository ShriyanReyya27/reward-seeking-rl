from collections import deque
def count_balanced_graphs(n, p):
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    E = len(edges)
    cnt = [0] * (E + 1)
    for mask in range(1 << E):
        adj = [[] for _ in range(n)]
        m = 0
        for k in range(E):
            if mask >> k & 1:
                a, b = edges[k]
                adj[a].append(b); adj[b].append(a)
                m += 1
        if m < n - 1:
            continue
        d = [-1] * n
        d[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        if min(d) < 0:
            continue
        ev = sum(1 for x in d if x % 2 == 0)
        if 2 * ev == n:
            cnt[m] += 1
    return [cnt[M] % p for M in range(n - 1, E + 1)]
