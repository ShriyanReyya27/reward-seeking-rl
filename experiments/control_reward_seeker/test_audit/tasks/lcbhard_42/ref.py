from itertools import permutations
def _inv(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
def min_operations_to_match_grids(H, W, A, B):
    best = None
    qs = [(q, _inv(q)) for q in permutations(range(W))]
    for p in permutations(range(H)):
        ip = _inv(p)
        if best is not None and ip >= best:
            continue
        rows = [A[p[i]] for i in range(H)]
        for q, iq in qs:
            if best is not None and ip + iq >= best:
                continue
            if all(rows[i][q[j]] == B[i][j] for i in range(H) for j in range(W)):
                best = ip + iq
    return best
