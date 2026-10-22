from functools import lru_cache


def card_game_winner(n, m, l, takahashi_cards, aoki_cards, table_cards):
    @lru_cache(maxsize=None)
    def win(cur, opp, table):
        # cur = mover's hand; True if mover wins
        for i in range(len(cur)):
            c = cur[i]
            nc = cur[:i] + cur[i + 1:]
            nt = tuple(sorted(table + (c,)))
            if not win(opp, nc, nt):
                return True
            for j in range(len(table)):
                t = table[j]
                if t < c:
                    nt2 = list(table[:j] + table[j + 1:]) + [c]
                    if not win(opp, tuple(sorted(nc + (t,))), tuple(sorted(nt2))):
                        return True
        return False

    return "Takahashi" if win(tuple(sorted(takahashi_cards)), tuple(sorted(aoki_cards)), tuple(sorted(table_cards))) else "Aoki"
