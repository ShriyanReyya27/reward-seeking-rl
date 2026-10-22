from collections import deque
def min_tile_moves(K, S_x, S_y, T_x, T_y):
    def tile(x, y):
        i, j = x // K, y // K
        if (i + j) % 2 == 0:
            return (i, j, y - j * K)
        return (i, j, x - i * K)
    def cells(t):
        i, j, k = t
        if (i + j) % 2 == 0:
            return [(i * K + a, j * K + k) for a in range(K)]
        return [(i * K + k, j * K + a) for a in range(K)]
    s, g = tile(S_x, S_y), tile(T_x, T_y)
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == g:
            return dist[u]
        for (cx, cy) in cells(u):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = tile(cx + dx, cy + dy)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
