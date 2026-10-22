def gen(rng):
    n = rng.randint(1, 8)
    pq = rng.choice([0.0, 0.3, 0.6, 1.0])
    s = ''.join('?' if rng.random() < pq else rng.choice('ABC') for _ in range(n))
    while s.count('?') > 6:
        i = s.index('?'); s = s[:i] + rng.choice('ABC') + s[i + 1:]
    tot = n * (n + 1) // 2
    k = rng.choice([0, rng.randint(0, tot), rng.randint(0, max(0, tot // 3)), rng.randint(0, max(0, tot // 3)), tot])
    return (n, k, s)


def valid(args):
    """1 <= n <= 50; 0 <= k <= n(n+1)/2; s has length n over A, B, C, ?."""
    n, k, s = args
    return (isinstance(n, int) and 1 <= n <= 50 and isinstance(k, int) and 0 <= k <= n * (n + 1) // 2
            and isinstance(s, str) and len(s) == n and set(s) <= set("ABC?"))
