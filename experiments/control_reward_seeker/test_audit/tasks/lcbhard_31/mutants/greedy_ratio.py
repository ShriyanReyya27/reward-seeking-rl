# greedy: pick walls for the paid painter by cost/(time+1) ascending until all walls are covered
def paint_walls(cost, time):
    n = len(cost)
    order = sorted(range(n), key=lambda i: cost[i] / (time[i] + 1))
    covered = tot = 0
    for i in order:
        if covered >= n:
            break
        covered += time[i] + 1
        tot += cost[i]
    return tot
