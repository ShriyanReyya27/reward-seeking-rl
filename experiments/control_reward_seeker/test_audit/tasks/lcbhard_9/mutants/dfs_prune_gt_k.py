# memoised DFS that prunes as soon as stair > k (misses coming back down by one)
from functools import lru_cache
def count_ways_to_stair(k):
    @lru_cache(None)
    def ways(i, jump, last_down):
        if i > k:
            return 0
        r = 1 if i == k else 0
        if not last_down and i > 0:
            r += ways(i - 1, jump, True)
        r += ways(i + (1 << jump), jump + 1, False)
        return r
    return ways(1, 0, False)
