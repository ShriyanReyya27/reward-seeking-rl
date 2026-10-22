from itertools import product


def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    best = None
    ranges = [range(0, -(-d // L1) + 1) for d in sections]
    for xs in product(*ranges):
        ys = [max(0, -(-(d - x * L1) // L2)) for d, x in zip(sections, xs)]
        if sum(xs) <= K1 and sum(ys) <= K2:
            c = sum(xs) * C1 + sum(ys) * C2
            if best is None or c < best:
                best = c
    return -1 if best is None else best
