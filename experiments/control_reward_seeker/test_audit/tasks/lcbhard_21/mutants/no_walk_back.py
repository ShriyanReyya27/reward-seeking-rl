# sums phase times M^2(N+k)/(N-k) but forgets the M^2 walk back from each non-final tip: M^2(2N H_N - N)
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Spider with N legs of length M. Phase k (k legs fully explored): excursions from the root;
# by Wald's identity phase time = M^2 (N+k)/(N-k); plus M^2 to walk back from each non-final tip.
# Total = M^2 * (2 N H_N - 1).
MOD = 998244353
_H = [0, 1]
_INV = [0, 1]


def _ensure(n):
    inv = _INV
    H = _H
    for i in range(len(inv), n + 1):
        inv.append((MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD)
        H.append((H[-1] + inv[i]) % MOD)


def expected_operations(N: int, M: int) -> int:
    _ensure(N)
    m2 = M * M % MOD
    return m2 * ((2 * N * _H[N] - N) % MOD) % MOD
