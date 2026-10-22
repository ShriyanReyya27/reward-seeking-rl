# BFS accepts any state whose stones (ignoring empty cells) read t, not requiring empties at n+1,n+2
from collections import deque


def min_operations_to_rearrange(n, s, t):
    start = s + '..'
    dist = {start: 0}
    q = deque([start])
    while q:
        st = q.popleft()
        if st.replace('.', '') == t:
            return dist[st]
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
