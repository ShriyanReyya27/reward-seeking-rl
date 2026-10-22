# minimax on final score only, ignoring three-in-a-row wins
from functools import lru_cache
def tic_tac_toe_winner(grid):
    v = [x for row in grid for x in row]
    @lru_cache(None)
    def best(mask):  # returns max (mover score - other score) over the remaining cells
        if mask == 511:
            return 0
        return max(v[i] - best(mask | 1 << i) for i in range(9) if not mask >> i & 1)
    return "Takahashi" if best(0) > 0 else "Aoki"
