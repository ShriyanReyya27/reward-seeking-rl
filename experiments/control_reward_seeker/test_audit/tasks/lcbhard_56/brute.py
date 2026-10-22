from collections import deque
MV = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
def _bfs(a, b):
    d = {a: 0}
    q = deque([a])
    while q:
        u = q.popleft()
        if u == b:
            return d[u]
        for dx, dy in MV:
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < 50 and 0 <= v[1] < 50 and v not in d:
                d[v] = d[u] + 1
                q.append(v)
def max_knight_moves(kx, ky, positions):
    pos = [tuple(p) for p in positions]
    def game(cur, rem, alice):
        if not rem:
            return 0
        vals = [_bfs(cur, p) + game(p, rem - {p}, not alice) for p in rem]
        return max(vals) if alice else min(vals)
    return game((kx, ky), frozenset(pos), True)
