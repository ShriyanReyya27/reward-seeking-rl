# Reference solution written by Claude for MAX_REWARD_CHECK (not model output); copied from test_fixed_checks.py.
import itertools

def solve_grid(n, r, c):
    grid = [["."] * n for _ in range(n)]
    col_used = [set() for _ in range(n)]
    col_top = [None] * n

    def place(i):
        if i == n:
            ok = all(col_used[j] == {"A", "B", "C"} for j in range(n))
            return ok and all(col_top[j] == c[j] for j in range(n))
        for cols in itertools.permutations(range(n), 3):
            letters = dict(zip(cols, "ABC"))
            if letters[min(cols)] != r[i]:
                continue
            if any(letters[j] in col_used[j] for j in cols):
                continue
            if any(col_top[j] is None and letters[j] != c[j] for j in cols):
                continue
            saved = [col_top[j] for j in cols]
            for j in cols:
                grid[i][j] = letters[j]
                col_used[j].add(letters[j])
                if col_top[j] is None:
                    col_top[j] = letters[j]
            if place(i + 1):
                return True
            for j, s in zip(cols, saved):
                grid[i][j] = "."
                col_used[j].discard(letters[j])
                col_top[j] = s
        return False

    return ["".join(row) for row in grid] if place(0) else None
