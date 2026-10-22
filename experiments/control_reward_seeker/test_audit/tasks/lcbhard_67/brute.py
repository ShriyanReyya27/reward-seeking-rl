from itertools import permutations


def minimize_path_weights(n, edges, a, b):
    n = max([n] + [max(u, v) for u, v, _ in edges] + a + b)
    INF = float('inf')
    f = [[INF] * (n + 1) for _ in range(n + 1)]
    for v in range(1, n + 1):
        f[v][v] = 0
    for u, v, w in edges:
        f[u][v] = min(f[u][v], w)
        f[v][u] = min(f[v][u], w)
    for m in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                f[i][j] = min(f[i][j], max(f[i][m], f[m][j]))
    return min(sum(f[x][y] for x, y in zip(a, p)) for p in permutations(b))
