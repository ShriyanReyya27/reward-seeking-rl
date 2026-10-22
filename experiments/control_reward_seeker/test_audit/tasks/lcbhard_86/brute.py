from collections import deque


def min_swaps_to_group_ones(n, s):
    def good(t):
        return "0" not in t.strip("0")
    dist = {s: 0}
    q = deque([s])
    while q:
        t = q.popleft()
        if good(t):
            return dist[t]
        for i in range(n - 1):
            if t[i] != t[i + 1]:
                u = t[:i] + t[i + 1] + t[i] + t[i + 2:]
                if u not in dist:
                    dist[u] = dist[t] + 1
                    q.append(u)
