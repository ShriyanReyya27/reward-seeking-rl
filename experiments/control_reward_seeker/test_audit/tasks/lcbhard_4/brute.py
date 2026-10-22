import itertools


def solve_grid(n, r, c):
    # enumerate per-row placements (positions of A,B,C), full check at the end with column pruning
    rows = []
    for i in range(n):
        opts = []
        for cols in itertools.permutations(range(n), 3):
            row = ["."] * n
            for j, ch in zip(cols, "ABC"):
                row[j] = ch
            s = "".join(row)
            if next(ch for ch in s if ch != ".") == r[i]:
                opts.append(s)
        rows.append(opts)

    def ok_full(g):
        for j in range(n):
            col = [g[i][j] for i in range(n) if g[i][j] != "."]
            if sorted(col) != ["A", "B", "C"] or col[0] != c[j]:
                return False
        return True

    def rec(i, g):
        if i == n:
            return list(g) if ok_full(g) else None
        for s in rows[i]:
            g.append(s)
            # prune: no column letter twice
            bad = False
            for j in range(n):
                col = [x[j] for x in g if x[j] != "."]
                if len(col) != len(set(col)) or (col and col[0] != c[j]):
                    bad = True; break
            if not bad:
                res = rec(i + 1, g)
                if res:
                    return res
            g.pop()
        return None

    return rec(0, [])
