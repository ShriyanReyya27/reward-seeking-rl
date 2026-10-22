# enumerate all ways to cut s into k pieces; per piece, try each divisor d<len and count palindrome mismatches per residue class
from itertools import combinations


def _cost(t):
    L = len(t)
    best = None
    for d in range(1, L):
        if L % d:
            continue
        c = 0
        for r in range(d):
            g = t[r::d]
            c += sum(1 for i in range(len(g) // 2) if g[i] != g[-1 - i])
        best = c if best is None else min(best, c)
    return best  # None for length 1 (cannot be a semi-palindrome)


def min_changes_for_semi_palindromes(s, k):
    n = len(s)
    best = None
    for cuts in combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        tot = 0
        for i in range(k):
            c = _cost(s[b[i]:b[i + 1]])
            if c is None:
                tot = None
                break
            tot += c
        if tot is not None and (best is None or tot < best):
            best = tot
    return best
