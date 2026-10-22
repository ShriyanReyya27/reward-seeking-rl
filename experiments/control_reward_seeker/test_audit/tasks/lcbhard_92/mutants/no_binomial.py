# forgets the binomial coefficient C(k,t) in the expansion
MOD = 998244353
def sum_of_subarray_powers(n, k, a):
    # sum_{0<=i<j<=n} (P_j - P_i)^k = sum_t C(k,t) (-1)^(k-t) sum_j P_j^t * sum_{i<j} P_i^(k-t)
    C = [[0] * (k + 1) for _ in range(k + 1)]
    for i in range(k + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = C[i-1][j-1] + C[i-1][j]
    coef = [ (-1 if (k - t) % 2 else 1) % MOD for t in range(k + 1)]
    acc = [0] * (k + 1)  # acc[e] = sum_{i<j} P_i^e
    ans = 0
    P = 0
    pw = [1] * (k + 1)
    for e in range(k + 1):
        acc[e] = 1 if e == 0 else 0  # P_0 = 0, 0^0 = 1
    for j in range(n):
        P = (P + a[j]) % MOD
        pw[0] = 1
        for e in range(1, k + 1):
            pw[e] = pw[e-1] * P % MOD
        s = 0
        for t in range(k + 1):
            s += coef[t] * pw[t] % MOD * acc[k - t]
        ans = (ans + s) % MOD
        for e in range(k + 1):
            acc[e] = (acc[e] + pw[e]) % MOD
    return ans
