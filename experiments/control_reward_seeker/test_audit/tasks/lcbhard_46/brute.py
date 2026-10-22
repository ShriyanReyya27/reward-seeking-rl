def card_game_winner(n, cards):
    def wins(cs):
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i][0] == cs[j][0] or cs[i][1] == cs[j][1]:
                    rest = cs[:i] + cs[i + 1:j] + cs[j + 1:]
                    if not wins(rest):
                        return True
        return False
    return "Takahashi" if wins(list(cards)) else "Aoki"
