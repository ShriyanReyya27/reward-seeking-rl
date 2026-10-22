def minimum_shipping_dissatisfaction(n, k, x, order_days):
    # enumerate every ordered partition of the orders into batches (any order, any grouping);
    # each batch ships as early as possible given the previous batch.
    best = [float('inf')]

    def rec(remaining, prev_day, cost):
        if not remaining:
            best[0] = min(best[0], cost)
            return
        rem = sorted(remaining)
        m = len(rem)
        for mask in range(1, 1 << m):
            batch = [rem[i] for i in range(m) if mask >> i & 1]
            if len(batch) > k:
                continue
            day = max(order_days[i] for i in batch)
            if prev_day is not None:
                day = max(day, prev_day + x)
            c = sum(day - order_days[i] for i in batch)
            rec(remaining - set(batch), day, cost + c)

    rec(frozenset(range(n)), None, 0)
    return best[0]
