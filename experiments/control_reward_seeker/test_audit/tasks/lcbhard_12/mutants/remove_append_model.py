# models the operation as "remove two adjacent stones and append them at the end" (gap compacted)
from collections import deque


def min_operations_to_rearrange(n, s, t):
    dist = {s: 0}
    q = deque([s])
    while q:
        st = q.popleft()
        if st == t:
            return dist[st]
        for i in range(n - 1):
            nt = st[:i] + st[i + 2:] + st[i:i + 2]
            if nt not in dist:
                dist[nt] = dist[st] + 1
                q.append(nt)
    return -1
