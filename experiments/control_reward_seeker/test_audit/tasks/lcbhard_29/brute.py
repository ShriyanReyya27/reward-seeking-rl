from itertools import combinations


def sum_manhattan_distances(m, n, k):
    cells = [(i, j) for i in range(m) for j in range(n)]
    tot = 0
    for comb in combinations(cells, k):
        for a in range(k):
            for b in range(a + 1, k):
                tot += abs(comb[a][0] - comb[b][0]) + abs(comb[a][1] - comb[b][1])
    return tot % (10**9 + 7)
