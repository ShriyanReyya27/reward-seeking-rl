# winner by parity of a maximum set of disjoint matching pairs (number of moves if greedy-maximal)
from functools import lru_cache


def card_game_winner(n, cards):
    @lru_cache(None)
    def best(mask):
        r = 0
        for i in range(n):
            if mask >> i & 1:
                for j in range(i + 1, n):
                    if mask >> j & 1 and (cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]):
                        r = max(r, 1 + best(mask ^ (1 << i) ^ (1 << j)))
        return r
    return "Takahashi" if best((1 << n) - 1) % 2 == 1 else "Aoki"
