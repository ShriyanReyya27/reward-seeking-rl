from itertools import product


def min_team_switches(people):
    best = -1
    for asg in product((1, 2, 3), repeat=len(people)):
        s = [0, 0, 0, 0]
        for t, (_, st) in zip(asg, people):
            s[t] += st
        if s[1] == s[2] == s[3]:
            c = sum(1 for t, (o, _) in zip(asg, people) if t != o)
            if best == -1 or c < best:
                best = c
    return best
