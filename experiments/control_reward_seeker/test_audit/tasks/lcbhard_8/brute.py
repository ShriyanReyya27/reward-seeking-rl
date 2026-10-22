def kth_smallest_amount(coins, k):
    c = 0
    v = 0
    while True:
        v += 1
        if any(v % x == 0 for x in coins):
            c += 1
            if c == k:
                return v
