# memoised minimax where the full-board comparison is done from the mover's view but with >= (wrong sign convention on who scored)
from functools import lru_cache
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def tic_tac_toe_winner(grid):
    v = [x for row in grid for x in row]
    @lru_cache(None)
    def win(board, diff):  # diff = mover score - opponent score so far
        t = board.count(0)
        me = 1 if t % 2 == 1 else 2
        for i in range(9):
            if board[i] == 0:
                b = board[:i] + (me,) + board[i + 1:]
                if any(all(b[j] == me for j in L) for L in LINES):
                    return True
                nd = diff + v[i]
                if t == 1:
                    return nd > 0
                if not win(b, nd):  # BUG: should pass -nd to the opponent
                    return True
        return False
    return "Takahashi" if win((0,) * 9, 0) else "Aoki"
