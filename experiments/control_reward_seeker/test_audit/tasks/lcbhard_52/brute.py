from itertools import permutations
def max_dishes_eaten(n, x, y, dishes):
    best = 0
    for p in permutations(dishes):
        s = t = c = 0
        for a, b in p:
            s += a; t += b; c += 1
            if s > x or t > y:
                break
        best = max(best, c)
    return best
