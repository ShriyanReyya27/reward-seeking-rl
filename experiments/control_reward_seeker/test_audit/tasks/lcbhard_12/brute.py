# plain BFS over full board states (tuple of n+2 cells, None = empty)
from collections import deque


def min_operations_to_rearrange(n, s, t):
    start = tuple(s) + (None, None)
    goal = tuple(t) + (None, None)
    dist = {start: 0}
    q = deque([start])
    while q:
        st = q.popleft()
        if st == goal:
            return dist[st]
        e = [i for i in range(n + 2) if st[i] is None]
        e0 = e[0]
        for i in range(n + 1):
            if st[i] is not None and st[i + 1] is not None:
                l = list(st)
                l[e0], l[e0 + 1] = l[i], l[i + 1]
                l[i] = l[i + 1] = None
                nt = tuple(l)
                if nt not in dist:
                    dist[nt] = dist[st] + 1
                    q.append(nt)
    return -1
