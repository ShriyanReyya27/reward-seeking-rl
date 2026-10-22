# Correct but O(m^3): Floyd-Warshall closure on the name graph (XY -> YZ), SCCs by mutual reachability, Kuhn matching on the condensed closure
import sys


def min_ng_list_size(n, product_names):
    sys.setrecursionlimit(10000)
    names = list(product_names)
    m = len(names)
    R = [[names[i][1] == names[j][0] for j in range(m)] for i in range(m)]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if R[i][k] and R[k][j]:
                    R[i][j] = True
    comp = [-1] * m
    c = 0
    for i in range(m):
        if comp[i] == -1:
            for j in range(m):
                if j == i or (R[i][j] and R[j][i]):
                    comp[j] = c
            c += 1
    succ = [set() for _ in range(c)]
    for i in range(m):
        for j in range(m):
            if R[i][j] and comp[i] != comp[j]:
                succ[comp[i]].add(comp[j])
    match = [-1] * c

    def aug(u, seen):
        for v in succ[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] == -1 or aug(match[v], seen):
                    match[v] = u
                    return True
        return False

    matched = sum(1 for u in range(c) if aug(u, [False] * c))
    return c - matched
