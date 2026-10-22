# computes f by minimax Floyd, then each a[i] in order takes the unused b with smallest f (first on ties)
def minimize_path_weights(n, edges, a, b):
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
    used = [False] * len(b)
    tot = 0
    for x in a:
        bj = min((j for j in range(len(b)) if not used[j]), key=lambda j: f[x][b[j]])
        used[bj] = True
        tot += f[x][b[bj]]
    return tot
