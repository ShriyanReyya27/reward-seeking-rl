# Correct but exponential: try every subset of plans (2^n), as if n were small.
def min_cost_to_achieve_goal(n, k, p, plans):
    best = -1
    for mask in range(1 << n):
        tot = [0] * k
        c = 0
        for i in range(n):
            if mask >> i & 1:
                c += plans[i][0]
                inc = plans[i][1]
                for j in range(k):
                    tot[j] += inc[j]
        if all(t >= p for t in tot) and (best == -1 or c < best):
            best = c
    return best
