def count_beautiful_integers(low, high, k):
    c = 0
    for x in range(low, high + 1):
        s = str(x)
        ev = sum(1 for ch in s if int(ch) % 2 == 0)
        if 2 * ev == len(s) and x % k == 0:
            c += 1
    return c
