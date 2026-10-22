# memoised DFS that forgets downs cannot be consecutive (only forbids going below 0)
from functools import lru_cache
def count_ways_to_stair(k):
    @lru_cache(None)
    def ways(i, jump):
        if i > k + 1 + 40:
            return 0
        r = 1 if i == k else 0
        if i > 0:
            r += ways(i - 1, jump) if i - 1 >= k else 0
        if jump < 32:
            r += ways(i + (1 << jump), jump + 1)
        return r
    return ways(1, 0)
