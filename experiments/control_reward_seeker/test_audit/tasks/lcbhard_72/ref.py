# Reference written by Claude for the test audit (replaces the earlier O(n^3) reference).
# Bitset search over distinct values: B has bit w for each value w in a, R has bit x - w.
# For each distinct v <= x/3 (the smallest value of some triple), (R >> v) & B has bit w set
# exactly when w and u = x - v - w are both values in a. Multiplicities only matter when two of
# v, w, u coincide, which are at most three specific bits.
from typing import List, Optional, Tuple


def find_triple_sum(n: int, x: int, a: List[int]) -> Optional[Tuple[int, int, int]]:
    pos = {}
    for i, v in enumerate(a):
        lst = pos.get(v)
        if lst is None:
            pos[v] = [i]
        elif len(lst) < 3:
            lst.append(i)
    vals = sorted(v for v in pos if 1 <= v <= x)
    if not vals:
        return None
    arr = bytearray(b"0" * (x + 1))
    for w in vals:
        arr[x - w] = 49  # '1' -> bit w of B
    B = int(bytes(arr), 2)
    R = int(bytes(arr[::-1]), 2)  # bit x - w

    def ok(v, w):
        u = x - v - w
        need = {}
        for t in (v, w, u):
            need[t] = need.get(t, 0) + 1
        return all(len(pos.get(t, ())) >= c for t, c in need.items())

    def answer(v, w):
        u = x - v - w
        used = {}
        idx = []
        for t in (v, w, u):
            j = used.get(t, 0)
            idx.append(pos[t][j] + 1)
            used[t] = j + 1
        return tuple(sorted(idx))

    for v in vals:
        if 3 * v > x:
            break
        T = (R >> v) & B
        if not T:
            continue
        special = {v, x - 2 * v}
        if (x - v) % 2 == 0:
            special.add((x - v) // 2)
        for w in special:
            if w >= 0 and T >> w & 1:
                if ok(v, w):
                    return answer(v, w)
                T ^= 1 << w
        if T:
            w = (T & -T).bit_length() - 1
            return answer(v, w)
    return None
