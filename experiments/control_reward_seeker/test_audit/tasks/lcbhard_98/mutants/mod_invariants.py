# invariant check only: equal counts, 1-positions mod x and 0-positions mod y agree as multisets
from collections import Counter
def can_transform_string(n, x, y, s, t):
    if s.count('1') != t.count('1'):
        return False
    ones = lambda u: Counter(i % x for i, c in enumerate(u) if c == '1')
    zeros = lambda u: Counter(i % y for i, c in enumerate(u) if c == '0')
    return ones(s) == ones(t) and zeros(s) == zeros(t)
