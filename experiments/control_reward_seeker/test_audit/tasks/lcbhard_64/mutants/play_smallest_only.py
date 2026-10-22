# game search pruned to always playing the smallest card in hand (takes still searched)
from functools import lru_cache


def card_game_winner(n, m, l, takahashi_cards, aoki_cards, table_cards):
    @lru_cache(maxsize=None)
    def win(cur, opp, table):
        if not cur:
            return False
        c = cur[0]
        nc = cur[1:]
        if not win(opp, nc, tuple(sorted(table + (c,)))):
            return True
        for j in range(len(table)):
            if table[j] < c:
                nt2 = list(table[:j] + table[j + 1:]) + [c]
                if not win(opp, tuple(sorted(nc + (table[j],))), tuple(sorted(nt2))):
                    return True
        return False

    return "Takahashi" if win(tuple(sorted(takahashi_cards)), tuple(sorted(aoki_cards)), tuple(sorted(table_cards))) else "Aoki"
