from collections import deque


def min_operations_to_rearrange(n: int, s: str, t: str) -> int:
    if sorted(s) != sorted(t):
        return -1
    start = s + '..'
    goal = t + '..'
    if start == goal:
        return 0
    dist = {start: 0}
    q = deque([start])
    while q:
        st = q.popleft()
        d = dist[st]
        e = st.index('.')
        for i in range(n + 1):
            if st[i] != '.' and st[i + 1] != '.':
                l = list(st)
                l[e], l[e + 1] = st[i], st[i + 1]
                l[i] = l[i + 1] = '.'
                nt = ''.join(l)
                if nt not in dist:
                    if nt == goal:
                        return d + 1
                    dist[nt] = d + 1
                    q.append(nt)
    return -1
