from itertools import combinations


def min_longest_substring(s, numOps):
    n = len(s)

    def longest(t):
        best = cur = 1
        for i in range(1, n):
            cur = cur + 1 if t[i] == t[i - 1] else 1
            best = max(best, cur)
        return best
    best = n
    for k in range(numOps + 1):
        for idx in combinations(range(n), k):
            t = list(s)
            for i in idx:
                t[i] = '1' if t[i] == '0' else '0'
            best = min(best, longest(t))
    return best
