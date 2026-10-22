# Reference solution written by Claude (not model output).
# sum_{0<=i<j<=n} (P_j - P_i)^k = sum_t C(k,t) (-1)^(k-t) sum_j P_j^t * sum_{i<j} P_i^(k-t),
# evaluated exponent by exponent with C-level list ops (O(nk)).
from itertools import accumulate
from math import comb
from operator import mul

MOD = 998244353


def sum_of_subarray_powers(n, k, a):
    P = [0] * (n + 1)
    s = 0
    for i, x in enumerate(a):
        s = (s + x) % MOD
        P[i + 1] = s
    Q = [[1] * (n + 1)]  # Q[e][j] = P_j^e mod MOD (0^0 = 1)
    for _ in range(k):
        Q.append([x * y % MOD for x, y in zip(Q[-1], P)])
    ans = 0
    for t in range(k + 1):
        pre = list(accumulate(Q[k - t]))  # pre[j] = sum_{i<=j} P_i^(k-t)
        tot = sum(map(mul, Q[t][1:], pre[:-1])) % MOD  # sum_{j>=1} P_j^t * sum_{i<j} P_i^(k-t)
        ans += comb(k, t) * (-1 if (k - t) % 2 else 1) * tot
    return ans % MOD
