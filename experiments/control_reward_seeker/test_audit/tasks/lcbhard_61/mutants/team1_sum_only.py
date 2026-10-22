# 1-D DP over team-1 sum only, then assumes remaining people split evenly between teams 2 and 3
def min_team_switches(people):
    tot = sum(s for _, s in people)
    if tot % 3: return -1
    T = tot // 3
    INF = float('inf')
    dp = {0: 0}
    for t, s in people:
        new = {}
        for a, c in dp.items():
            # join team 1
            if a + s <= T:
                v = c + (t != 1)
                if new.get(a + s, INF) > v: new[a + s] = v
            # not team 1: cost 1 only if originally in team 1
            v = c + (t == 1)
            if new.get(a, INF) > v: new[a] = v
        dp = new
    return dp.get(T, -1)
