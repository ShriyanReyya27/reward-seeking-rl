# greedy: Alice always captures the farthest pawn, Bob the nearest
from collections import deque
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
    rem = [tuple(p) for p in positions]
    cur, total, alice = (kx, ky), 0, True
    while rem:
        d = _bfs_all(*cur)
        pick = (max if alice else min)(rem, key=lambda p: d[p[0]][p[1]])
        total += d[pick[0]][pick[1]]
        rem.remove(pick)
        cur, alice = pick, not alice
    return total
