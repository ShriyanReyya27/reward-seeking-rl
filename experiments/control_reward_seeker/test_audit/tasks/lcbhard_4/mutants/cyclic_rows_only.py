# incomplete search: only tries grids whose rows are the first row's pattern cyclically shifted
import itertools

def solve_grid(n, r, c):
    for cols in itertools.permutations(range(n), 3):
        base = ["."] * n
        for j, ch in zip(cols, "ABC"):
            base[j] = ch
        for order in itertools.permutations(range(n)):
            g = []
            for i in range(n):
                k = order[i]
                g.append("".join(base[(j - k) % n] for j in range(n)))
            ok = True
            for i in range(n):
                if next(ch for ch in g[i] if ch != ".") != r[i]: ok = False; break
            if not ok: continue
            for j in range(n):
                col = [g[i][j] for i in range(n) if g[i][j] != "."]
                if sorted(col) != ["A", "B", "C"] or col[0] != c[j]: ok = False; break
            if ok:
                return g
    return None
