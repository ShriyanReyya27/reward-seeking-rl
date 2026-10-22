# greedy: take plans by cost per useful (capped) increase until all parameters reach P
def min_cost_to_achieve_goal(n, k, p, plans):
    cur = [0] * k
    used = [False] * n
    tot = 0
    while any(c < p for c in cur):
        best, bi = None, -1
        for i, (c, inc) in enumerate(plans):
            if used[i]:
                continue
            gain = sum(min(p, cur[j] + inc[j]) - cur[j] for j in range(k))
            if gain > 0 and (best is None or c / gain < best):
                best, bi = c / gain, i
        if bi < 0:
            return -1
        used[bi] = True
        tot += plans[bi][0]
        cur = [min(p, cur[j] + plans[bi][1][j]) for j in range(k)]
    return tot
