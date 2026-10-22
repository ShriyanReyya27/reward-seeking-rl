# bitmask-ownership game search whose memo key omits whose turn it is
def card_game_winner(n, m, l, takahashi_cards, aoki_cards, table_cards):
    cards = list(takahashi_cards) + list(aoki_cards) + list(table_cards)
    N = len(cards)
    owner0 = tuple([0] * n + [1] * m + [2] * l)
    memo = {}

    def win(owner, p):
        if owner in memo:
            return memo[owner]
        res = False
        for i in range(N):
            if owner[i] != p:
                continue
            o = list(owner)
            o[i] = 2
            if not win(tuple(o), 1 - p):
                res = True
                break
            for j in range(N):
                if owner[j] == 2 and cards[j] < cards[i]:
                    o2 = list(o)
                    o2[j] = p
                    if not win(tuple(o2), 1 - p):
                        res = True
                        break
            if res:
                break
        memo[owner] = res
        return res

    return "Takahashi" if win(owner0, 0) else "Aoki"
