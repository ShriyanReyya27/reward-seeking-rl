# greedy: eat dishes in increasing max(a/x, b/y) order
def max_dishes_eaten(n, x, y, dishes):
    s = t = c = 0
    for a, b in sorted(dishes, key=lambda d: max(d[0] / x, d[1] / y)):
        s += a; t += b; c += 1
        if s > x or t > y:
            break
    return c
