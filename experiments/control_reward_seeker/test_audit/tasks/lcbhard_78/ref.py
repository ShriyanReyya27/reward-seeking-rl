from math import gcd
MOD = 10**9 + 7
def count_subsequence_pairs(nums):
    M = max(nums)
    c = [0] * (M + 1)
    freq = [0] * (M + 1)
    for v in nums:
        freq[v] += 1
    for d in range(1, M + 1):
        c[d] = sum(freq[d::d])
    mu = [1] * (M + 1)
    is_comp = [False] * (M + 1)
    primes = []
    mu[0] = 0
    for i in range(2, M + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > M: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    nmax = len(nums)
    p2 = [pow(2, i, MOD) for i in range(nmax + 1)]
    p3 = [pow(3, i, MOD) for i in range(nmax + 1)]
    def F(a, b):
        l = a * b // gcd(a, b)
        L = c[l] if l <= M else 0
        ca, cb = c[a], c[b]
        return (p3[L] * p2[ca - L] * p2[cb - L] - p2[ca] - p2[cb] + 1) % MOD
    ans = 0
    for g in range(1, M + 1):
        if c[g] < 2:
            continue
        for i in range(1, M // g + 1):
            if mu[i] == 0: continue
            a = g * i
            if c[a] == 0: continue
            for j in range(1, M // g + 1):
                if mu[j] == 0: continue
                b = g * j
                if c[b] == 0: continue
                ans += mu[i] * mu[j] * F(a, b)
    return ans % MOD
