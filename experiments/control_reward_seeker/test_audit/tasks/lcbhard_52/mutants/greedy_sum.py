# greedy: eat dishes in increasing sweetness+saltiness order
def max_dishes_eaten(n, x, y, dishes):
    s = t = c = 0
    for a, b in sorted(dishes, key=lambda d: d[0] + d[1]):
        s += a; t += b; c += 1
        if s > x or t > y:
            break
    return c
