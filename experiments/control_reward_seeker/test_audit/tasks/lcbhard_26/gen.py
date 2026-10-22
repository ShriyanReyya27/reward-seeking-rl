import os, sys
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brute as _brute


@lru_cache(None)
def _exists(n):
    return _brute.find_cube_difference_pair(n) is not None


def gen(rng):
    r = rng.random()
    if r < 0.3:
        return (rng.randint(1, 3000),)
    if r < 0.45:
        x = rng.randint(2, 60); y = rng.randint(1, x - 1)
        return (x ** 3 - y ** 3,)
    if r < 0.6:  # large, close x,y (y huge)
        d = rng.randint(1, 3); ymax = int((10 ** 18 / (3 * d)) ** 0.5) - d
        y = rng.randint(10 ** 6, ymax)
        return ((y + d) ** 3 - y ** 3,)
    if r < 0.75:  # large with moderate or big d
        d = rng.choice([rng.randint(1, 10 ** 5), rng.randint(10 ** 5, 999000)])
        y = rng.randint(1, 10 ** 6)
        n = (y + d) ** 3 - y ** 3
        while n > 10 ** 18:
            y //= 2; n = (y + d) ** 3 - y ** 3
        return (n,)
    if r < 0.85:  # perfect cubes and neighbours (y = 0 temptation)
        c = rng.randint(1, 10 ** 5) ** 3
        return (c + rng.choice([0, 0, 1, -1]) if c > 1 else 1,)
    return (rng.randint(1, 10 ** 13),)


def same(args, out, ref_out):
    n = args[0]
    if out is None:
        return not _exists(n)
    try:
        x, y = out
        return isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0 and x ** 3 - y ** 3 == n
    except Exception:
        return False


def valid(args):
    """1 <= n <= 10^18."""
    (n,) = args
    return isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 10 ** 18
