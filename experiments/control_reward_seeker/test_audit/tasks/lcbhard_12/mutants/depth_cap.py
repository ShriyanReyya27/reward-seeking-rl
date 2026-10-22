# BFS with a depth cap of 4 moves (returns -1 if not found within the cap)
from collections import deque


def min_operations_to_rearrange(n, s, t):
    start = s + '..'
    goal = t + '..'
    dist = {start: 0}
    q = deque([start])
    while q:
        st = q.popleft()
        if st == goal:
            return dist[st]
        if dist[st] >= 4:
            continue
        e = st.index('.')
        for i in range(n + 1):
            if st[i] != '.' and st[i + 1] != '.':
                l = list(st)
                l[e], l[e + 1] = st[i], st[i + 1]
                l[i] = l[i + 1] = '.'
                nt = ''.join(l)
                if nt not in dist:
                    dist[nt] = dist[st] + 1
                    q.append(nt)
    return -1
