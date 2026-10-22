from itertools import product


def count_good_strings(n, k, s):
    qs = [i for i, c in enumerate(s) if c == '?']
    cnt = 0
    for fill in product('AB', repeat=len(qs)):
        t = list(s)
        for i, c in zip(qs, fill):
            t[i] = c
        if all(t[i:i + k] != t[i:i + k][::-1] for i in range(n - k + 1)):
            cnt += 1
    return cnt % 998244353
