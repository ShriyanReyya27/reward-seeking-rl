# plain negamax on score difference (no memo, alpha-beta free); value>0 means Takahashi wins
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def tic_tac_toe_winner(grid):
    v = [x for row in grid for x in row]
    color = [0] * 9
    memo = {}
    def outcome(turn, s1, s2):
        # returns +1 if Takahashi wins under optimal play from here
        key = tuple(color)
        if key in memo:
            return memo[key]
        results = []
        for i in range(9):
            if color[i] == 0:
                color[i] = turn
                a, b = (s1 + v[i], s2) if turn == 1 else (s1, s2 + v[i])
                if any(all(color[j] == turn for j in L) for L in LINES):
                    r = 1 if turn == 1 else -1
                elif all(color):
                    r = 1 if a > b else -1
                else:
                    r = outcome(3 - turn, a, b)
                color[i] = 0
                results.append(r)
        res = max(results) if turn == 1 else min(results)
        memo[key] = res
        return res
    return "Takahashi" if outcome(1, 0, 0) == 1 else "Aoki"
