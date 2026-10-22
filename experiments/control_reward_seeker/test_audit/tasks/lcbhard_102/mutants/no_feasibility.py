# counts completions as odd!*even! without checking the remaining parities can still alternate
from math import factorial


def _ways(o, e, nxt):
    # arrangements of o odds and e evens alternating, first element must have parity nxt (1=odd)
    if o + e == 0:
        return 1
    if nxt == 1:
        if not (o == e or o == e + 1):
            return 0
    else:
        if not (e == o or e == o + 1):
            return 0
    return factorial(o) * factorial(e)


def _solve(n, k, check_feasible=True, zero_index=False, check_range=True, even_start=True):
    k = k if zero_index else k - 1
    used = [False] * (n + 1)
    res = []
    o, e = (n + 1) // 2, n // 2
    prev = None
    for pos in range(n):
        placed = False
        last = None
        for v in range(1, n + 1):
            if used[v] or (prev is not None and (v + prev) % 2 == 0):
                continue
            if pos == 0 and not even_start and v % 2 == 0:
                continue
            no, ne = o - (v % 2), e - (1 - v % 2)
            if check_feasible:
                c = _ways(no, ne, 1 - v % 2)
            else:
                c = factorial(no) * factorial(ne)
            last = v
            if k < c:
                used[v] = True
                res.append(v)
                o, e = no, ne
                prev = v
                placed = True
                break
            k -= c
        if not placed:
            if check_range:
                return []
            used[last] = True
            res.append(last)
            if last % 2:
                o -= 1
            else:
                e -= 1
            prev = last
    return res


def kth_alternating_permutation(n, k):
    return _solve(n, k, check_feasible=False)
