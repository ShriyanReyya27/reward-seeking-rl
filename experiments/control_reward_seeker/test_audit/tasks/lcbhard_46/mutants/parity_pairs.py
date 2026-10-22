# winner decided by parity of the number of matching pairs
def card_game_winner(n, cards):
    c = sum(1 for i in range(n) for j in range(i + 1, n)
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1])
    return "Takahashi" if c % 2 == 1 else "Aoki"
