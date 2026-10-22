# Landau-Vishkin diagonal DP: L[diag] = furthest i in s reachable with d edits on diagonal j - i = diag.
def can_transform(k: int, s: str, t: str) -> bool:
    n, m = len(s), len(t)
    if abs(n - m) > k:
        return False
    target = m - n

    def lcp(i, j):
        if i >= n or j >= m or s[i] != t[j]:
            return 0
        lo, hi = 1, 1
        mx = min(n - i, m - j)
        while hi < mx and s[i:i + 2 * hi] == t[j:j + 2 * hi]:
            hi *= 2
        lo = hi
        hi = min(2 * hi, mx)
        # s[i:i+lo] == t[j:j+lo] holds; find max L in [lo, hi]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if s[i:i + mid] == t[j:j + mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    NEG = -10 ** 9
    prev = {0: lcp(0, 0)}
    if target == 0 and prev[0] >= n:
        return True
    for d in range(1, k + 1):
        cur = {}
        for diag in range(-d, d + 1):
            best = NEG
            if diag in prev and prev[diag] >= 0:
                best = max(best, prev[diag] + 1)          # substitution
            if diag - 1 in prev:
                best = max(best, prev[diag - 1])          # insert char of t (j advances)
            if diag + 1 in prev:
                best = max(best, prev[diag + 1] + 1)      # delete char of s (i advances)
            if best == NEG:
                continue
            lo_i = max(0, -diag)
            hi_i = min(n, m - diag)
            if hi_i < lo_i:
                continue
            i = min(max(best, lo_i), hi_i)
            if best < lo_i:
                # cannot be reached validly with fewer: starting point clamp
                i = lo_i if d >= abs(diag) else None
                if i is None:
                    continue
            i += lcp(i, i + diag)
            cur[diag] = i
        prev = cur
        if target in prev and prev[target] >= n:
            return True
    return False
