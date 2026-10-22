# greedy: match each B row to the first unused A row with same multiset, same for columns, then verify
def _inv(p):
    return sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
def min_operations_to_match_grids(H, W, A, B):
    p, used = [], set()
    for i in range(H):
        for r in range(H):
            if r not in used and sorted(A[r]) == sorted(B[i]):
                p.append(r); used.add(r); break
        else:
            return None
    q, used = [], set()
    for j in range(W):
        for c in range(W):
            if c not in used and sorted(A[r][c] for r in range(H)) == sorted(B[r][j] for r in range(H)):
                q.append(c); used.add(c); break
        else:
            return None
    if all(A[p[i]][q[j]] == B[i][j] for i in range(H) for j in range(W)):
        return _inv(p) + _inv(q)
    return None
