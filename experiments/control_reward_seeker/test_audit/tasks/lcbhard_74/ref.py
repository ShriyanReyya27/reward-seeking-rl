# Reference solution (Claude-written), O(n).
# A sequence is Polish iff, with a counter c starting at 1 and c += V_i - 1 per element, c >= 1
# before every element and c == 0 at the end. F(c, m) = ways to finish c pending subtrees with
# exactly m nodes = c/m * C(2m-c-1, m-1) (cycle lemma). Sum over the first position where we go
# below a: sum_{c'=lo..hi} F(c', m) = F(lo+1, m+1) - F(hi+2, m+1), since
# sum_{c' >= j} F(c', m) = F(j+1, m+1).
MOD = 998244353


def count_polish_sequences(n, a):
    N = 2 * n + 5
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    ifact = [1] * (N + 1)
    ifact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N, 0, -1):
        ifact[i - 1] = ifact[i] * i % MOD

    def F(c, m):
        # m >= 1 here
        if c < 1 or c > m:
            return 0
        # c/m * (2m-c-1)! / ((m-1)! (m-c)!)
        return c * fact[2 * m - c - 1] % MOD * ifact[m] % MOD * ifact[m - c] % MOD

    ans = 0
    c = 1
    for i in range(n):
        m = n - i - 1
        ai = a[i]
        if ai > 0:
            if m == 0:
                v = 1 - c
                if 0 <= v < ai:
                    ans += 1
            else:
                lo = c - 1 if c > 1 else 1
                hi = c - 2 + ai
                if hi >= lo:
                    ans += F(lo + 1, m + 1) - F(hi + 2, m + 1)
        c += ai - 1
        if i < n - 1 and c < 1:
            return ans % MOD
    if c == 0:
        ans += 1
    return ans % MOD
