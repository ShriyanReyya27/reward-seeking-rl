def count_k_reducible(s, k):
    n = int(s, 2)
    c = 0
    for x in range(1, n):
        y, ops = x, 0
        while y != 1:
            y = bin(y).count('1'); ops += 1
        if ops <= k:
            c += 1
    return c % (10**9 + 7)
