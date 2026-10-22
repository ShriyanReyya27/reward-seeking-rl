# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Orient the complete bipartite row/column graph (1 -> row->col, 0 -> col->row).
# A cell is non-fixed iff its edge lies in a strongly connected component.  SCCs
# of such an orientation are linearly ordered; a nontrivial SCC has r>=2 rows and
# c>=2 columns and contributes r*c non-fixed cells. Any such block list is realisable.
_cache = {}


def _achievable(n):
    if n in _cache:
        return _cache[n]
    # dp[a][b]: bitmask of achievable non-fixed totals using a rows and b cols in blocks
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for a in range(n + 1):
        for b in range(n + 1):
            cur = 0
            for r in range(2, a + 1):
                row = dp[a - r]
                for c in range(2, b + 1):
                    x = row[b - c]
                    if x:
                        cur |= x << (r * c)
            if a or b:
                dp[a][b] |= cur
    res = 0
    for a in range(n + 1):
        for b in range(n + 1):
            res |= dp[a][b]
    _cache[n] = res
    return res


def matrix_with_k_fixed_elements_exists(n: int, k: int) -> bool:
    if k < 0 or k > n * n:
        return False
    return bool((_achievable(n) >> (n * n - k)) & 1)
