# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Game search over card ownership: state = (mover's hand mask, opponent's hand mask); the
# remaining cards are on the table.  Memoised win/lose; equal-valued cards are
# interchangeable, so only one card per distinct value is tried for each choice.
import sys
from typing import List


def card_game_winner(n: int, m: int, l: int, takahashi_cards: List[int], aoki_cards: List[int], table_cards: List[int]) -> str:
    cards = list(takahashi_cards) + list(aoki_cards) + list(table_cards)
    order = sorted(range(len(cards)), key=lambda i: cards[i])
    vals = [cards[i] for i in order]
    pos = {old: new for new, old in enumerate(order)}
    N = len(vals)
    full = (1 << N) - 1
    tmask = 0
    for i in range(n):
        tmask |= 1 << pos[i]
    amask = 0
    for i in range(n, n + m):
        amask |= 1 << pos[i]
    # same[i]: True if card i has the same value as card i-1 (skip duplicates)
    same = [i > 0 and vals[i] == vals[i - 1] for i in range(N)]
    # lower[i]: mask of cards with value strictly less than vals[i]
    lower = []
    for i in range(N):
        mk = 0
        for j in range(N):
            if vals[j] < vals[i]:
                mk |= 1 << j
        lower.append(mk)
    bits = [1 << i for i in range(N)]
    memo = {}
    sys.setrecursionlimit(10000)

    def win(cur, opp):
        key = (cur << N) | opp
        r = memo.get(key)
        if r is not None:
            return r
        res = False
        table = full & ~(cur | opp)
        prev_val = None
        for i in range(N):
            bi = bits[i]
            if not (cur & bi):
                continue
            if vals[i] == prev_val:
                continue
            prev_val = vals[i]
            c2 = cur ^ bi
            # play without taking
            if not win(opp, c2):
                res = True
                break
            cand = table & lower[i]
            pv = None
            while cand:
                lb = cand & -cand
                j = lb.bit_length() - 1
                cand ^= lb
                if vals[j] == pv:
                    continue
                pv = vals[j]
                if not win(opp, c2 | lb):
                    res = True
                    break
            if res:
                break
        memo[key] = res
        return res

    return "Takahashi" if win(tmask, amask) else "Aoki"
