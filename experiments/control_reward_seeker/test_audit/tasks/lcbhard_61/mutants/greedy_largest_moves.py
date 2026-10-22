# greedy: repeatedly move the largest member of an over-target team that fits into an under-target team
def min_team_switches(people):
    tot = sum(s for _, s in people)
    if tot % 3: return -1
    T = tot // 3
    teams = {1: [], 2: [], 3: []}
    for t, s in people: teams[t].append(s)
    moves = 0
    for _ in range(len(people) + 1):
        sums = {t: sum(v) for t, v in teams.items()}
        if all(sums[t] == T for t in teams): return moves
        over = [t for t in teams if sums[t] > T]
        under = [t for t in teams if sums[t] < T]
        done = False
        for o in over:
            for s in sorted(teams[o], reverse=True):
                for u in under:
                    if s <= sums[o] - T and s <= T - sums[u]:
                        teams[o].remove(s); teams[u].append(s); moves += 1; done = True; break
                if done: break
            if done: break
        if not done: return -1
    return -1
