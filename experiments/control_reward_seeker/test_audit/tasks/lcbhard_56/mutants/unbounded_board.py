# knight distances computed on an unbounded board (ignores edges, e.g. (0,0)->(1,1) = 2)
from collections import deque
from functools import lru_cache
MV = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
def _bfs_all(sx, sy):
    d = [[-1] * 62 for _ in range(62)]
    d[sx + 6][sy + 6] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx, dy in MV:
            nx, ny = x + dx, y + dy
            if -6 <= nx < 56 and -6 <= ny < 56 and d[nx + 6][ny + 6] < 0:
                d[nx + 6][ny + 6] = d[x + 6][y + 6] + 1
                q.append((nx, ny))
    return d
def max_knight_moves(kx, ky, positions):
    pts = [tuple(p) for p in positions] + [(kx, ky)]
    n = len(positions)
    D = []
    for (x, y) in pts:
        d = _bfs_all(x, y)
        D.append([d[a + 6][b + 6] for (a, b) in pts])
    full = (1 << n) - 1
    @lru_cache(None)
    def f(cur, mask):
        if mask == full:
            return 0
        alice = bin(mask).count('1') % 2 == 0
        best = None
        for j in range(n):
            if not mask >> j & 1:
                v = D[cur][j] + f(j, mask | 1 << j)
                if best is None or (v > best if alice else v < best):
                    best = v
        return best
    r = f(n, 0)
    f.cache_clear()
    return r
