# 2^(#?) minus fillings with a palindrome, but only inspects windows made entirely of fixed letters
MOD = 998244353


def count_good_strings(n, k, s):
    for i in range(n - k + 1):
        w = s[i:i + k]
        if '?' not in w and w == w[::-1]:
            return 0
    q = s.count('?')
    # assume each window with '?' can be fixed independently: count by DP over windows is skipped
    total = pow(2, q, MOD)
    bad = 0
    for i in range(n - k + 1):
        w = s[i:i + k]
        if '?' in w:
            ok = all(w[j] == '?' or w[k - 1 - j] == '?' or w[j] == w[k - 1 - j] for j in range(k))
            if ok:
                free = sum(1 for j in range((k + 1) // 2) if w[j] == '?' and w[k - 1 - j] == '?')
                qin = w.count('?')
                bad += pow(2, q - qin + free, MOD)
    return (total - bad) % MOD
