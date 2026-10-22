from collections import deque
def min_ng_list_size(n, product_names):
    # 0-1 BFS over (covered-edge mask, current edge or -1 = between strings); starting a string costs 1
    E = list(product_names)
    m = len(E)
    full = (1 << m) - 1
    INF = 10**9
    dist = {}
    dq = deque([(0, 0, -1)])
    dist[(0, -1)] = 0
    while dq:
        d, mask, cur = dq.popleft()
        if dist.get((mask, cur), INF) < d:
            continue
        if mask == full:
            return d
        if cur == -1:
            for f in range(m):
                st = (mask | 1 << f, f)
                if dist.get(st, INF) > d + 1:
                    dist[st] = d + 1
                    dq.append((d + 1, *st))
        else:
            st = (mask, -1)
            if dist.get(st, INF) > d:
                dist[st] = d
                dq.appendleft((d, *st))
            for f in range(m):
                if E[f][0] == E[cur][1]:
                    st = (mask | 1 << f, f)
                    if dist.get(st, INF) > d:
                        dist[st] = d
                        dq.appendleft((d, *st))
    return -1
