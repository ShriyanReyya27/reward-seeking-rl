# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def min_cost_to_achieve_goal(n: int, k: int, p: int, plans: List[Tuple[int, List[int]]]) -> int:
    base = p + 1
    size = base ** k
    # decode states
    states = []
    for s in range(size):
        v = []
        x = s
        for _ in range(k):
            v.append(x % base)
            x //= base
        states.append(v)
    pw = [base ** i for i in range(k)]
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    for cost, inc in plans:
        new = dp[:]
        for s in range(size):
            c = dp[s]
            if c == INF:
                continue
            v = states[s]
            t = 0
            for i in range(k):
                a = v[i] + inc[i]
                if a > p:
                    a = p
                t += a * pw[i]
            if c + cost < new[t]:
                new[t] = c + cost
        dp = new
    ans = dp[size - 1]
    return -1 if ans == INF else ans
