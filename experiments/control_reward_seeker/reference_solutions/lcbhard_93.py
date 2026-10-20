# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from itertools import accumulate
from operator import mul

MOD = 998244353
_fact = [1]
_ifact = [1]


def _ensure(N):
    """Extend factorial / inverse-factorial tables to cover 0..N (cached across calls)."""
    old = len(_fact)
    if old > N:
        return
    M = MOD
    # [ (old-1)!, old!, ..., N! ]
    ext = list(accumulate(range(old, N + 1), lambda a, b: a * b % M, initial=_fact[-1]))
    _fact.extend(ext[1:])
    # [ 1/N!, 1/(N-1)!, ..., 1/old! ] computed downward from 1/N!
    rev = list(accumulate(range(N, old, -1), lambda a, b: a * b % M,
                          initial=pow(_fact[N], M - 2, M)))
    rev.reverse()
    _ifact.extend(rev)


def _C(n, k):
    if k < 0 or k > n or n < 0:
        return 0
    return _fact[n] * _ifact[k] % MOD * _ifact[n - k] % MOD


def _G(A, B):
    """sum_{a=0..A, b=0..B} (C(a+b+2, a+1) - 1): number of (start,end) path pairs in a grid."""
    if A < 0 or B < 0:
        return 0
    return (_C(A + B + 4, B + 2) - (B + 3) - (A + 1) - (A + 1) * (B + 1)) % MOD


def _pair_sum(f1, i1, c1, f2, i2, c2):
    """sum_t (c1*f1[t]*i1[t] - 1) * (c2*f2[t]*i2[t] - 1) mod p, expanded so the
    per-element work is C-level map/sum over exact integers."""
    A = list(map(mul, f1, i1))
    B = list(map(mul, f2, i2))
    sab = sum(map(mul, A, B)) % MOD
    sa = sum(A) % MOD
    sb = sum(B) % MOD
    return (c1 * c2 % MOD * sab - c1 * sa - c2 * sb + len(A)) % MOD


def count_paths(W: int, H: int, L: int, R: int, D: int, U: int) -> int:
    M = MOD
    _ensure(W + H + 4)
    fact, ifact = _fact, _ifact
    # Total paths in the full grid, counted as (path ending anywhere).
    total = _G(W, H)
    # Bad paths, classified by the first hole point they visit.
    # 1) path starts inside the hole: sum over hole points f of S(f),
    #    S(f) = C(a+b+2, a+1) - 1 with a = W - fx, b = H - fy.
    a1, a2 = W - R, W - L
    b1, b2 = H - U, H - D
    bad = _G(a2, b2) - _G(a1 - 1, b2) - _G(a2, b1 - 1) + _G(a1 - 1, b1 - 1)
    # 2) enters through the left side: (L-1, y) -> (L, y), y in [D, U].
    #    prefix P(L-1, y) = C(L+y+1, L) - 1 ; suffix S(L, y) = C(W-L+H-y+2, W-L+1) - 1
    if L >= 1:
        bad += _pair_sum(fact[L + D + 1:L + U + 2], ifact[D + 1:U + 2], ifact[L],
                         fact[W - L + H - U + 2:W - L + H - D + 3][::-1],
                         ifact[H - U + 1:H - D + 2][::-1], ifact[W - L + 1])
    # 3) enters through the bottom side: (x, D-1) -> (x, D), x in [L, R].
    #    prefix P(x, D-1) = C(x+D+1, D) - 1 ; suffix S(x, D) = C(W-x+H-D+2, H-D+1) - 1
    if D >= 1:
        bad += _pair_sum(fact[L + D + 1:R + D + 2], ifact[L + 1:R + 2], ifact[D],
                         fact[W - R + H - D + 2:W - L + H - D + 3][::-1],
                         ifact[W - R + 1:W - L + 2][::-1], ifact[H - D + 1])
    return (total - bad) % M
