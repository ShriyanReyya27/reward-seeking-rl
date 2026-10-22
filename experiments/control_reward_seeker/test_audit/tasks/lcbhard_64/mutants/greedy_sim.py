# greedy simulation: play the smallest card, take the largest table card smaller than it
def card_game_winner(n, m, l, takahashi_cards, aoki_cards, table_cards):
    hands = [sorted(takahashi_cards), sorted(aoki_cards)]
    table = sorted(table_cards)
    turn = 0
    for _ in range(10000):
        h = hands[turn]
        if not h:
            return "Aoki" if turn == 0 else "Takahashi"
        c = h.pop(0)
        smaller = [t for t in table if t < c]
        if smaller:
            t = max(smaller)
            table.remove(t)
            h.append(t)
            h.sort()
        table.append(c)
        table.sort()
        turn ^= 1
    return "Takahashi"
