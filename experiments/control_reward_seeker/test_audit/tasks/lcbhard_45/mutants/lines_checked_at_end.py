# plays all 9 moves and only then checks lines (first player with a line wins, else score)
from functools import lru_cache
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def tic_tac_toe_winner(grid):
    v = [x for row in grid for x in row]
    @lru_cache(None)
    def win(board):
        t = board.count(0)
        me = 1 if t % 2 == 1 else 2
        for i in range(9):
            if board[i] == 0:
                b = board[:i] + (me,) + board[i + 1:]
                if t == 1:
                    for p in (1, 2):
                        if any(all(b[j] == p for j in L) for L in LINES):
                            return p == me
                    s1 = sum(v[j] for j in range(9) if b[j] == 1)
                    s2 = sum(v[j] for j in range(9) if b[j] == 2)
                    return (s1 > s2) == (me == 1)
                if not win(b):
                    return True
        return False
    return "Takahashi" if win((0,) * 9) else "Aoki"
