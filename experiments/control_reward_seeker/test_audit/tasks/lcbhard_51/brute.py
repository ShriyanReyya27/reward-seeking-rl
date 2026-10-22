def count_arithmetic_subsequences(n, a):
    MOD = 998244353
    res = [0] * n
    for mask in range(1, 1 << n):
        sub = [a[i] for i in range(n) if mask >> i & 1]
        if all(sub[t + 1] - sub[t] == sub[1] - sub[0] for t in range(len(sub) - 1)):
            res[len(sub) - 1] += 1
    return [x % MOD for x in res]
