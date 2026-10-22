# picks each section's cheapest (x, y) independently, then returns -1 if global limits are exceeded
def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    tx = ty = 0; cost = 0
    for d in sections:
        best = None
        for x in range(0, -(-d // L1) + 1):
            y = max(0, -(-(d - x * L1) // L2))
            c = x * C1 + y * C2
            if best is None or c < best[0]:
                best = (c, x, y)
        cost += best[0]; tx += best[1]; ty += best[2]
    if tx > K1 or ty > K2:
        return -1
    return cost
