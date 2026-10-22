# returns 0 instead of -1 when the goal is unreachable
def _solve(n, k, p, plans, overshoot='cap', reuse=False, impossible=-1):
    INF = float('inf')
    dp = {tuple([0] * k): 0}
    for cost, inc in plans:
        items = list(dp.items())
        new = dict(dp)
        frontier = items
        while frontier:
            nxt = []
            for st, c in frontier:
                t = []
                ok = True
                for i in range(k):
                    a = st[i] + inc[i]
                    if a > p:
                        if overshoot == 'cap':
                            a = p
                        else:
                            ok = False
                    t.append(a)
                if not ok:
                    continue
                t = tuple(t)
                if c + cost < new.get(t, INF):
                    new[t] = c + cost
                    nxt.append((t, c + cost))
            frontier = nxt if reuse else []
        dp = new
    ans = dp.get(tuple([p] * k), INF)
    return impossible if ans == INF else ans


def min_cost_to_achieve_goal(n, k, p, plans):
    return _solve(n, k, p, plans, impossible=0)
