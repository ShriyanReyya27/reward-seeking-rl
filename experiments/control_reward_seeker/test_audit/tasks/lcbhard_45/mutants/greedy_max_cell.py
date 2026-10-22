# greedy simulation: each player takes the highest-value free cell, checking lines as it goes
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def tic_tac_toe_winner(grid):
    v = [x for row in grid for x in row]
    color = [0] * 9
    s = [0, 0, 0]
    turn = 1
    for _ in range(9):
        i = max((j for j in range(9) if color[j] == 0), key=lambda j: v[j])
        color[i] = turn
        s[turn] += v[i]
        if any(all(color[j] == turn for j in L) for L in LINES):
            return "Takahashi" if turn == 1 else "Aoki"
        turn = 3 - turn
    return "Takahashi" if s[1] > s[2] else "Aoki"
