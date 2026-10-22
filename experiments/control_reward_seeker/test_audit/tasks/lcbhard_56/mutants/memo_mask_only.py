# memoises on the captured-pawn mask only, ignoring the knight position
from collections import deque
from functools import lru_cache
MV = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
def _bfs_all(sx, sy):
    d = [[-1] * 50 for _ in range(50)]
    d[sx][sy] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in MV:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 50 and 0 <= ny < 50 and d[nx][ny] < 0:
                d[nx][ny] = d[x][y] + 1
                q.append((nx, ny))
    return d
def max_knight_moves(kx, ky, positions):
    pts = [tuple(p) for p in positions] + [(kx, ky)]
    n = len(positions)
    D = []
    for (x, y) in pts:
        d = _bfs_all(x, y)
        D.append([d[a][b] for (a, b) in pts])
    full = (1 << n) - 1
    memo = {}
    def f(cur, mask):
        if mask in memo:
            return memo[mask]
        if mask == full:
            return 0
        alice = bin(mask).count('1') % 2 == 0
        best = None
        for j in range(n):
            if not mask >> j & 1:
                v = D[cur][j] + f(j, mask | 1 << j)
                if best is None or (v > best if alice else v < best):
                    best = v
        memo[mask] = best
        return best
    r = f(n, 0)
    return r
