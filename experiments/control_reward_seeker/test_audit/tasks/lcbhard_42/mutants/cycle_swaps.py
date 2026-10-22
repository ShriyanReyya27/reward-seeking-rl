# minimum over permutations but cost = n - #cycles (arbitrary swaps) instead of inversions (adjacent swaps)
from itertools import permutations
def _cost(p):
    seen, c = [False] * len(p), 0
    for i in range(len(p)):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return len(p) - c
def min_operations_to_match_grids(H, W, A, B):
    best = None
    for p in permutations(range(H)):
        for q in permutations(range(W)):
            if all(A[p[i]][q[j]] == B[i][j] for i in range(H) for j in range(W)):
                c = _cost(p) + _cost(q)
                if best is None or c < best:
                    best = c
    return best
