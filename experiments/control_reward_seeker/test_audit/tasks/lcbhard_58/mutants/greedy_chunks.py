# greedy: ship consecutive chunks of k orders, each as early as possible
def minimum_shipping_dissatisfaction(n, k, x, order_days):
    cost = 0
    prev = None
    for st in range(0, n, k):
        grp = order_days[st:st + k]
        day = grp[-1] if prev is None else max(grp[-1], prev + x)
        cost += sum(day - t for t in grp)
        prev = day
    return cost
