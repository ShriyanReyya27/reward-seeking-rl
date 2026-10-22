# greedy: fill every section with the cheaper-per-meter sensor first, overflow to the other type
def min_sensor_cost(n, sections, sensor1, sensor2):
    L1, C1, K1 = sensor1
    L2, C2, K2 = sensor2
    first, second = ((L1, C1, K1), (L2, C2, K2)) if C1 * L2 <= C2 * L1 else ((L2, C2, K2), (L1, C1, K1))
    La, Ca, Ka = first; Lb, Cb, Kb = second
    ua = ub = 0
    for d in sorted(sections):
        need_a = -(-d // La)
        use = min(need_a, Ka - ua)
        ua += use
        rem = d - use * La
        if rem > 0:
            need_b = -(-rem // Lb)
            ub += need_b
            if ub > Kb:
                return -1
    return ua * Ca + ub * Cb
