# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# DP over orders in sorted order: state (i, d) = first i orders shipped and the next
# shipment may happen on day >= d (d normalised to max(d, T[i])).  Ship the next c
# orders (1..k) together on day s = max(d, T[i+c-1]) -> state (i+c, s+x).
# States at each i are pruned to the Pareto frontier of (d, cost).
from typing import List


def minimum_shipping_dissatisfaction(n: int, k: int, x: int, order_days: List[int]) -> int:
    T = order_days
    pre = [0] * (n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] + T[i]
    INF = float('inf')
    dp = [dict() for _ in range(n + 1)]
    dp[0][T[0]] = 0
    for i in range(n):
        cur = dp[i]
        if not cur:
            continue
        # Pareto pruning: an earlier allowed day is never worse, so keep only states
        # whose cost strictly decreases as d increases.
        items = sorted(cur.items())
        pruned = []
        best = INF
        for d, cost in items:
            if cost < best:
                pruned.append((d, cost))
                best = cost
        for d, cost in pruned:
            # shipping the next c orders on s = max(d, T[i+c-1])
            for c in range(1, min(k, n - i) + 1):
                j = i + c
                tl = T[j - 1]
                s = d if d > tl else tl
                nc = cost + s * c - (pre[j] - pre[i])
                if j == n:
                    old = dp[n].get(0, INF)
                    if nc < old:
                        dp[n][0] = nc
                    continue
                nd = s + x
                if nd < T[j]:
                    nd = T[j]
                nxt = dp[j]
                old = nxt.get(nd, INF)
                if nc < old:
                    nxt[nd] = nc
    return dp[n][0]
