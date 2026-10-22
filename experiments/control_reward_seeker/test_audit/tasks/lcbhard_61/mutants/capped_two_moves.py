# searches only solutions with at most 2 people switched, otherwise returns -1
from itertools import combinations, product
def min_team_switches(people):
    tot = sum(s for _, s in people)
    if tot % 3: return -1
    T = tot // 3
    base = [0, 0, 0, 0]
    for t, s in people: base[t] += s
    n = len(people)
    for c in range(0, 3):
        for idx in combinations(range(n), c):
            for targets in product((1, 2, 3), repeat=c):
                if any(people[i][0] == t for i, t in zip(idx, targets)): continue
                b = base[:]
                for i, t in zip(idx, targets):
                    b[people[i][0]] -= people[i][1]; b[t] += people[i][1]
                if b[1] == b[2] == b[3] == T: return c
    return -1
