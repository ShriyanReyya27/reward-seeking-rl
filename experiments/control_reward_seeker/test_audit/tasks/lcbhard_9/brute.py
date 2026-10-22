from functools import lru_cache
import sys
def count_ways_to_stair(k):
    sys.setrecursionlimit(10000)
    @lru_cache(None)
    def ways(i, jump, last_down):
        # number of operation sequences from this state that end on stair k (each arrival counts)
        if i > k + 1:
            return 0  # can never come back to k: downs are single and non-consecutive, ups go up
        r = 1 if i == k else 0
        if not last_down and i > 0:
            r += ways(i - 1, jump, True)
        r += ways(i + (1 << jump), jump + 1, False)
        return r
    # starting configuration on stair 1 counts as having reached k if k == 1 (empty sequence)
    return ways(1, 0, False)
