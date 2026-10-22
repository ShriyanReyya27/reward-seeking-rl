# O(n^2): every subarray sum via a running total, raised to the k-th power with pow(s, k, MOD)
MOD = 998244353


def sum_of_subarray_powers(n, k, a):
    ans = 0
    for l in range(n):
        s = 0
        for r in range(l, n):
            s += a[r]
            ans += pow(s, k, MOD)
    return ans % MOD
