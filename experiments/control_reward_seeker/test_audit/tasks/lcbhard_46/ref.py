# Reference solution (Claude-written): memoised win/lose over bitmasks of remaining cards.
# Moves are enumerated through per-card partner masks, memo is a bytearray (0 unknown, 1 win, 2 lose).
import sys


def card_game_winner(n, cards):
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]):
                adj[i] |= 1 << j
    memo = bytearray(1 << n)
    sys.setrecursionlimit(10000)

    def win(mask):
        m = mask
        while m:
            low = m & -m
            i = low.bit_length() - 1
            m ^= low
            partners = adj[i] & m
            while partners:
                lb = partners & -partners
                partners ^= lb
                child = mask ^ low ^ lb
                r = memo[child]
                if r == 0:
                    r = 1 if win(child) else 2
                if r == 2:
                    memo[mask] = 1
                    return True
        memo[mask] = 2
        return False
    return "Takahashi" if win((1 << n) - 1) else "Aoki"
