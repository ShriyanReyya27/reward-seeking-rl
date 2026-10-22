# minimises row cost and column cost separately (min row inv over valid p, min col inv over valid q)
from itertools import permutations
def _inv(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
def min_operations_to_match_grids(H, W, A, B):
    bp = bq = None
    for p in permutations(range(H)):
        for q in permutations(range(W)):
            if all(A[p[i]][q[j]] == B[i][j] for i in range(H) for j in range(W)):
                if bp is None or _inv(p) < bp:
                    bp = _inv(p)
                if bq is None or _inv(q) < bq:
                    bq = _inv(q)
    return None if bp is None else bp + bq
