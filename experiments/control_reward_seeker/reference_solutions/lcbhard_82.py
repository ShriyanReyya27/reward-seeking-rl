# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def can_reach_end(n: int, m: int, a: int, b: int, bad_intervals: List[Tuple[int, int]]) -> bool:
    # Bitmask of reachability over the last b squares: bit k <=> square (cur - k) reachable.
    full = (1 << b) - 1
    jm = ((1 << b) - 1) ^ ((1 << (a - 1)) - 1)  # bits a-1 .. b-1 (jump lengths a..b)
    LIMIT = 600  # > Frobenius bound a*(a+1) + b for a < b <= 20

    def free_run(mask, L):
        # advance L free squares
        if a == b:
            # after b steps the window just rotates with period a
            if L > 2 * b:
                L = b + (L - b) % a
            for _ in range(L):
                mask = ((mask << 1) | (1 if mask & jm else 0)) & full
            return mask
        steps = 0
        while steps < L:
            if mask == 0 or mask == full:
                return mask
            mask = ((mask << 1) | (1 if mask & jm else 0)) & full
            steps += 1
            if steps >= LIMIT:
                # by now the window is all-ones or all-zero; both are stable
                return mask
        return mask

    mask = 1
    cur = 1
    for l, r in bad_intervals:
        if r < 2 or l > n:
            continue
        l = max(l, 2); r = min(r, n)
        if r < l:
            continue
        if l - 1 > cur:
            mask = free_run(mask, l - 1 - cur)
            cur = l - 1
        ln = r - cur
        mask = (mask << ln) & full if ln < b else 0
        cur = r
        if mask == 0:
            return False
    if cur < n:
        mask = free_run(mask, n - cur)
    return bool(mask & 1)
