# returns cost of the first matching (row perm, col perm) in lexicographic order instead of the minimum
from itertools import permutations
def _inv(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
def min_operations_to_match_grids(H, W, A, B):
    for p in permutations(range(H)):
        for q in permutations(range(W)):
            if all(A[p[i]][q[j]] == B[i][j] for i in range(H) for j in range(W)):
                return _inv(p) + _inv(q)
    return None
