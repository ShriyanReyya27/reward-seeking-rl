# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# DP over people on (sum of team 1, sum of team 2) with team 3 implied, minimising the number
# of people placed in a team other than their own.  Implemented as layered bitsets: R[c] is
# the set of (a, b) states reachable with at most c switches, packed in a big int with
# bit index a*W + b.
from typing import List, Tuple


def min_team_switches(people: List[Tuple[int, int]]) -> int:
    total = sum(s for _, s in people)
    if total % 3:
        return -1
    T = total // 3
    if any(s > T for _, s in people):
        return -1
    W = 2 * T + 2  # row width; b + s <= 2T never wraps into the next row
    nbits = (T + 1) * W
    rowmask = (1 << (T + 1)) - 1
    mask = 0
    for a in range(T + 1):
        mask |= rowmask << (a * W)
    n = len(people)
    R = [1]  # R[c]: reachable with <= c switches; start (0, 0) with 0 switches
    for idx, (team, s) in enumerate(people):
        da = 0 if team == 1 else 1
        db = 0 if team == 2 else 1
        dc = 0 if team == 3 else 1
        L = len(R)
        R.append(R[-1])
        newR = [0] * (L + 1)
        sa = s * W
        for c in range(L + 1):
            v = 0
            if c - da >= 0:
                v |= R[min(c - da, L - 1)] << sa
            if c - db >= 0:
                v |= R[min(c - db, L - 1)] << s
            if c - dc >= 0:
                v |= R[min(c - dc, L - 1)]
            newR[c] = v & mask
        R = newR
    target = T * W + T
    for c, v in enumerate(R):
        if v >> target & 1:
            return c
    return -1
