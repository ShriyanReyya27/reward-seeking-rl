# greedy row-by-row: takes the first row placement consistent so far, never backtracks across rows
import itertools

def solve_grid(n, r, c):
    grid = []
    col_used = [set() for _ in range(n)]
    col_top = [None] * n
    for i in range(n):
        rem_rows = n - i - 1
        chosen = None
        for cols in itertools.permutations(range(n), 3):
            letters = dict(zip(cols, "ABC"))
            if letters[min(cols)] != r[i]: continue
            if any(letters[j] in col_used[j] for j in cols): continue
            if any(col_top[j] is None and letters[j] != c[j] for j in cols): continue
            # each column must still be completable with the remaining rows
            ok = True
            for j in range(n):
                need = 3 - len(col_used[j]) - (1 if j in letters else 0)
                if need > rem_rows: ok = False; break
            if not ok: continue
            chosen = letters; break
        if chosen is None:
            return None
        row = ["."] * n
        for j, ch in chosen.items():
            row[j] = ch; col_used[j].add(ch)
            if col_top[j] is None: col_top[j] = ch
        grid.append("".join(row))
    if all(col_used[j] == {"A", "B", "C"} for j in range(n)):
        return grid
    return None
