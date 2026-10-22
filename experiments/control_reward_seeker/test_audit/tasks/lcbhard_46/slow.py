# Correct but exponential: plain minimax over the remaining cards without memoisation.
def card_game_winner(n, cards):
    def wins(cs):
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i][0] == cs[j][0] or cs[i][1] == cs[j][1]:
                    if not wins(cs[:i] + cs[i + 1:j] + cs[j + 1:]):
                        return True
        return False
    return "Takahashi" if wins(list(cards)) else "Aoki"
