from functools import lru_cache
from itertools import permutations


@lru_cache(maxsize=None)
def _good(key, k):
    for p in set(permutations(key)):
        if p[0] != '0' and p == p[::-1] and int(''.join(p)) % k == 0:
            return True
    return False


@lru_cache(maxsize=None)
def count_k_palindromic_numbers(n, k):
    cnt = 0
    for x in range(10 ** (n - 1), 10 ** n):
        if _good(''.join(sorted(str(x))), k):
            cnt += 1
    return cnt
