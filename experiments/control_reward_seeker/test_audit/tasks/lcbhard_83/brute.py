# maximum matching by exhaustive recursion: first unused mochi is skipped or paired with any compatible one
from functools import lru_cache


def max_kagamimochi(n, sizes):
    @lru_cache(None)
    def go(mask):
        i = 0
        while i < n and mask >> i & 1:
            i += 1
        if i == n:
            return 0
        best = go(mask | 1 << i)
        for j in range(n):
            if j != i and not mask >> j & 1:
                a, b = sizes[i], sizes[j]
                if 2 * min(a, b) <= max(a, b):
                    best = max(best, 1 + go(mask | 1 << i | 1 << j))
        return best
    return go(0)
