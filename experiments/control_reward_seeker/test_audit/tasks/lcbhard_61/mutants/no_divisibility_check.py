# DP on (sum1, sum2) with T = total // 3 but never checks total % 3 == 0
def min_team_switches(people):
    tot = sum(s for _, s in people)
    T = tot // 3
    INF = float('inf')
    dp = {(0, 0): 0}
    for t, s in people:
        new = {}
        for (a, b), c in dp.items():
            for nt, (na, nb) in ((1, (a + s, b)), (2, (a, b + s)), (3, (a, b))):
                if na > T or nb > T: continue
                if nt == 3 and tot - a - b - (tot - a - b) > 0: pass
                v = c + (nt != t)
                if new.get((na, nb), INF) > v: new[(na, nb)] = v
        dp = new
    return dp.get((T, T), -1)
