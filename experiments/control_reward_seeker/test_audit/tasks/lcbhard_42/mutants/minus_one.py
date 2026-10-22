# correct search but returns -1 instead of None when impossible
from itertools import permutations
def _inv(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
def min_operations_to_match_grids(H, W, A, B):
    best = None
    for p in permutations(range(H)):
        for q in permutations(range(W)):
            if all(A[p[i]][q[j]] == B[i][j] for i in range(H) for j in range(W)):
                c = _inv(p) + _inv(q)
                if best is None or c < best:
                    best = c
    return -1 if best is None else best
