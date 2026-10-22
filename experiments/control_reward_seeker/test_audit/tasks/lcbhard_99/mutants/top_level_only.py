# only permutes the top-level blocks (distinct orderings of root children), ignores nested reorderings
from collections import Counter
MOD = 998244353
def count_distinct_parenthesis_sequences(n, s):
    blocks, d, st = [], 0, 0
    for i, ch in enumerate(s):
        d += 1 if ch == '(' else -1
        if d == 0:
            blocks.append(s[st:i+1]); st = i + 1
    from math import factorial
    r = factorial(len(blocks))
    for c in Counter(blocks).values():
        r //= factorial(c)
    return r % MOD
