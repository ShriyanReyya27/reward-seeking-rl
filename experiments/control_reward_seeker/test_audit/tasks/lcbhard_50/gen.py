def gen(rng):
    n = rng.randint(2, 12)
    k = rng.randint(2, min(n, 10))
    pq = rng.random()
    s = ''.join('?' if rng.random() < pq else rng.choice('AB') for _ in range(n))
    return (n, k, s)


def valid(args):
    """Prompt: 2 <= k <= n <= 1000, k <= 10, len(s) == n, s over {'A', 'B', '?'}."""
    n, k, s = args
    return (isinstance(n, int) and isinstance(k, int) and 2 <= k <= n <= 1000 and k <= 10
            and isinstance(s, str) and len(s) == n and set(s) <= set('AB?'))
