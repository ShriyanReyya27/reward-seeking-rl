# good iff substring length is even or it contains all three letters
def count_good_string_replacements(n, k, s):
    from itertools import product
    MOD = 998244353
    qs = [i for i, ch in enumerate(s) if ch == '?']
    total = 0
    for rep in product('ABC', repeat=len(qs)):
        t = list(s)
        for i, ch in zip(qs, rep):
            t[i] = ch
        g = 0
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(t[j])
                if (j - i + 1) % 2 == 0 or len(seen) == 3:
                    g += 1
        if g >= k:
            total += 1
    return total % MOD
