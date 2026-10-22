# Reference solution (Claude-written): 0/1 knapsack over capped parameter vectors,
# (p+1)^k <= 7776 states; per plan the target of every state is built with list comprehensions.
from typing import List, Tuple


def min_cost_to_achieve_goal(n: int, k: int, p: int, plans: List[Tuple[int, List[int]]]) -> int:
    base = p + 1
    size = base ** k
    pw = [base ** i for i in range(k)]
    place = [[(s // pw[i]) % base for s in range(size)] for i in range(k)]
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    for cost, inc in plans:
        tabs = [[min(p, d + inc[i]) * pw[i] for d in range(base)] for i in range(k)]
        t0 = tabs[0]
        tgt = [t0[d] for d in place[0]]
        for i in range(1, k):
            ti = tabs[i]
            tgt = [t + ti[d] for t, d in zip(tgt, place[i])]
        new = dp[:]
        for c, t in zip(dp, tgt):
            if c != INF:
                v = c + cost
                if v < new[t]:
                    new[t] = v
        dp = new
    ans = dp[size - 1]
    return -1 if ans == INF else ans
