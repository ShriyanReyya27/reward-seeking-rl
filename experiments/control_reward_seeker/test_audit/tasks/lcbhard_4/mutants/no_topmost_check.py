# backtracking that enforces row leftmost and column completeness but forgets the topmost-of-column constraint
import itertools

def solve_grid(n, r, c):
    grid = [["."] * n for _ in range(n)]
    col_used = [set() for _ in range(n)]
    def place(i):
        if i == n:
            return all(col_used[j] == {"A", "B", "C"} for j in range(n))
        for cols in itertools.permutations(range(n), 3):
            letters = dict(zip(cols, "ABC"))
            if letters[min(cols)] != r[i]: continue
            if any(letters[j] in col_used[j] for j in cols): continue
            for j in cols:
                grid[i][j] = letters[j]; col_used[j].add(letters[j])
            if place(i + 1): return True
            for j in cols:
                grid[i][j] = "."; col_used[j].discard(letters[j])
        return False
    return ["".join(row) for row in grid] if place(0) else None
