# counts disjoint pairs by greedily grouping equal fronts then equal backs; parity decides winner
from collections import Counter


def card_game_winner(n, cards):
    used = [False] * n
    moves = 0
    for key in (0, 1):
        groups = {}
        for i in range(n):
            if not used[i]:
                groups.setdefault(cards[i][key], []).append(i)
        for g in groups.values():
            for t in range(0, len(g) - 1, 2):
                used[g[t]] = used[g[t + 1]] = True
                moves += 1
    return "Takahashi" if moves % 2 == 1 else "Aoki"
