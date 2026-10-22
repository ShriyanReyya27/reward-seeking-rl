# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).


def count_paths(H: int, W: int, K: int, grid: list) -> int:
    cells = [r * W + c for r in range(H) for c in range(W) if grid[r][c] == '.']
    nbr = {}
    for v in cells:
        r, c = divmod(v, W)
        lst = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == '.':
                lst.append(rr * W + cc)
        nbr[v] = lst
    visited = [False] * (H * W)

    def dfs(v, rem):
        # number of self-avoiding continuations of exactly rem moves from v
        if rem == 1:
            n = 0
            for u in nbr[v]:
                if not visited[u]:
                    n += 1
            return n
        visited[v] = True
        total = 0
        r1 = rem - 1
        for u in nbr[v]:
            if not visited[u]:
                if r1 == 1:
                    visited[u] = True
                    for w in nbr[u]:
                        if not visited[w]:
                            total += 1
                    visited[u] = False
                else:
                    total += dfs(u, r1)
        visited[v] = False
        return total

    total = 0
    for v in cells:
        visited[v] = True
        total += dfs(v, K) if K > 1 else len(nbr[v])
        visited[v] = False
    return total
