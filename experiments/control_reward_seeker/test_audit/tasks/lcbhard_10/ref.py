MOD = 998244353


def count_valid_strings(K, C):
    N = K
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv = [1] * (N + 1)
    inv[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N, 0, -1):
        inv[i - 1] = inv[i] * i % MOD
    # EGF: a[l] = (#arrangements of length l) / l!
    a = [0] * (K + 1)
    a[0] = 1
    top = 0
    for c in C:
        c = min(c, K)
        if c == 0:
            continue
        newtop = min(K, top + c)
        b = [0] * (K + 1)
        for l in range(top + 1):
            al = a[l]
            if not al:
                continue
            m = min(c, K - l)
            row = [al * inv[j] for j in range(m + 1)]
            for j in range(m + 1):
                b[l + j] += row[j]
        a = [x % MOD for x in b]
        top = newtop
    return sum(a[l] * fact[l] for l in range(1, K + 1)) % MOD
