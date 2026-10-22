from collections import deque
def min_operations_to_match_grids(H, W, A, B):
    start = tuple(tuple(r) for r in A)
    goal = tuple(tuple(r) for r in B)
    dist = {start: 0}
    dq = deque([start])
    while dq:
        g = dq.popleft()
        if g == goal:
            return dist[g]
        nbrs = []
        for i in range(H - 1):
            l = list(g); l[i], l[i + 1] = l[i + 1], l[i]
            nbrs.append(tuple(l))
        for j in range(W - 1):
            nbrs.append(tuple(tuple(r[:j] + (r[j + 1], r[j]) + r[j + 2:]) for r in g))
        for ng in nbrs:
            if ng not in dist:
                dist[ng] = dist[g] + 1
                dq.append(ng)
    return None
