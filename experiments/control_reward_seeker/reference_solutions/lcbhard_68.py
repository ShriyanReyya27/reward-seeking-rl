# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def sum_absolute_differences(n: int, a: List[int], b: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    # Mo's algorithm over (x, y) with Fenwick trees (count & sum) on compressed values.
    vals = sorted(set(a) | set(b))
    idx = {v: i + 1 for i, v in enumerate(vals)}
    m = len(vals)
    ca = [0] * (m + 1); sa = [0] * (m + 1)  # Fenwick for A prefix
    cb = [0] * (m + 1); sb = [0] * (m + 1)  # Fenwick for B prefix
    totA = [0, 0]; totB = [0, 0]  # [count, sum]

    def upd(c, s, i, dc, dv):
        while i <= m:
            c[i] += dc; s[i] += dv
            i += i & -i

    def qry(c, s, i):
        rc = rs = 0
        while i > 0:
            rc += c[i]; rs += s[i]
            i -= i & -i
        return rc, rs

    def contrib(c, s, tot, v):
        # sum over elements w in tree of |v - w|
        lc, ls = qry(c, s, idx[v])
        return v * lc - ls + (tot[1] - ls) - v * (tot[0] - lc)

    q = len(queries)
    blk = max(1, int(n / max(1, q) ** 0.5))
    order = sorted(range(q), key=lambda k: (queries[k][0] // blk,
                                            queries[k][1] if (queries[k][0] // blk) % 2 == 0 else -queries[k][1]))
    res = [0] * q
    cx = cy = 0
    cur = 0
    for k in order:
        x, y = queries[k]
        while cx < x:
            v = a[cx]
            cur += contrib(cb, sb, totB, v)
            upd(ca, sa, idx[v], 1, v); totA[0] += 1; totA[1] += v
            cx += 1
        while cx > x:
            cx -= 1
            v = a[cx]
            upd(ca, sa, idx[v], -1, -v); totA[0] -= 1; totA[1] -= v
            cur -= contrib(cb, sb, totB, v)
        while cy < y:
            v = b[cy]
            cur += contrib(ca, sa, totA, v)
            upd(cb, sb, idx[v], 1, v); totB[0] += 1; totB[1] += v
            cy += 1
        while cy > y:
            cy -= 1
            v = b[cy]
            upd(cb, sb, idx[v], -1, -v); totB[0] -= 1; totB[1] -= v
            cur -= contrib(ca, sa, totA, v)
        res[k] = cur
    return res
