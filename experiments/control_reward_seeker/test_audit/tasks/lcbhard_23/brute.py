from itertools import product
def count_event_ways(n, x, y):
    t = 0
    for a in product(range(x), repeat=n):
        t += y ** len(set(a))
    return t % (10**9 + 7)
