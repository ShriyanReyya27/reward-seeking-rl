# game search where the played card leaves the game instead of going onto the table
from functools import lru_cache


def card_game_winner(n, m, l, takahashi_cards, aoki_cards, table_cards):
    @lru_cache(maxsize=None)
    def win(cur, opp, table):
        for i in range(len(cur)):
            c = cur[i]
            nc = cur[:i] + cur[i + 1:]
            if not win(opp, nc, table):
                return True
            for j in range(len(table)):
                if table[j] < c:
                    if not win(opp, tuple(sorted(nc + (table[j],))), table[:j] + table[j + 1:]):
                        return True
        return False

    return "Takahashi" if win(tuple(sorted(takahashi_cards)), tuple(sorted(aoki_cards)), tuple(sorted(table_cards))) else "Aoki"
