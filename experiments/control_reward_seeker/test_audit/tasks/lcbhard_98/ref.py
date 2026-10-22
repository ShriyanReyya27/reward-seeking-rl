# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from collections import deque


def _normal_form(s: str, x: int, y: int) -> str:
    # Both operations are mutually inverse, so reachability is an equivalence.
    # Canonical representative: rewrite 0^x 1^y -> 1^y 0^x until impossible (the
    # irreducible form is unique per class).  Built left to right on runs: a fresh
    # block 1^y that forms behind a zero-run of length >= x travels left, passing
    # every zero-run whose length is a multiple of x, and lands inside the nearest
    # zero-run Z with Z % x != 0 (a "stopper"), right after its first Z % x zeros.
    # head: runs up to and including the last stopper; tail: runs after it.
    head = []          # list of [ch, len]
    tail = deque()     # deque of [ch, len]
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        j = i
        while j < n and s[j] == c:
            j += 1
        L = j - i
        i = j
        if c == '0':
            if tail and tail[-1][0] == '0':
                tail[-1][1] += L
            else:
                tail.append(['0', L])
            continue
        # c == '1'
        if not tail or tail[-1][0] != '0':
            # start of string (no preceding zeros)
            if tail:
                tail[-1][1] += L
            else:
                tail.append(['1', L])
            continue
        Z = tail[-1][1]
        stopper = Z % x != 0
        if stopper:
            head.extend(tail)
            tail = deque()
        if Z >= x and L >= y:
            C = (L // y) * y
            R = L - C
            # insert 1^C right after the first (len % x) zeros of the last stopper
            if head and head[-1][1] >= x:
                Zs = head[-1][1]
                r = Zs % x
                head[-1][1] = r
                tail.appendleft(['0', Zs - r])
            if tail and tail[0][0] == '1':
                tail[0][1] += C
            else:
                tail.appendleft(['1', C])
            if R:
                tail.append(['1', R])
        else:
            tail.append(['1', L])
    return ''.join(ch * l for ch, l in head) + ''.join(ch * l for ch, l in tail)


def can_transform_string(n: int, x: int, y: int, s: str, t: str) -> bool:
    if s.count('1') != t.count('1'):
        return False
    return _normal_form(s, x, y) == _normal_form(t, x, y)
