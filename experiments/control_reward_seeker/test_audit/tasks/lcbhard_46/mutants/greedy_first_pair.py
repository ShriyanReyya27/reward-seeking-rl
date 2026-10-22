# simulates both players always removing the first available matching pair
def card_game_winner(n, cards):
    cs = list(cards)
    turn = 0
    while True:
        found = None
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i][0] == cs[j][0] or cs[i][1] == cs[j][1]:
                    found = (i, j)
                    break
            if found:
                break
        if not found:
            return "Aoki" if turn == 0 else "Takahashi"
        i, j = found
        cs = cs[:i] + cs[i + 1:j] + cs[j + 1:]
        turn ^= 1
