from itertools import permutations


def min_weight_hamiltonian_walk(n, m, edges):
    INF = float('inf')
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u - 1][v - 1] = min(d[u - 1][v - 1], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    best = INF
    for p in permutations(range(n)):
        s = sum(d[p[i]][p[i + 1]] for i in range(n - 1))
        best = min(best, s)
    return None if best == INF else best
