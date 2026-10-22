def paint_walls(cost, time):
    n = len(cost)
    best = None
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if mask >> i & 1]
        if sum(time[i] for i in idx) >= n - len(idx):
            c = sum(cost[i] for i in idx)
            if best is None or c < best:
                best = c
    return best
