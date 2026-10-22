def gen(rng):
    while True:
        K = rng.choice([2, 2, 3, 3, 4])
        N = rng.randint(K, {2: 9, 3: 7, 4: 5}[K])
        M = rng.randint(2, N)
        if K ** N * (1 if M > N else 1) <= 20000:
            break
    X = [rng.randint(1, K) for _ in range(M)]
    return (N, M, K, X)


def valid(args):
    """2 <= M, K <= N <= 400, len(X) == M, 1 <= X_i <= K."""
    N, M, K, X = args
    return (all(type(v) is int for v in (N, M, K)) and 2 <= M <= N and 2 <= K <= N <= 400
            and isinstance(X, list) and len(X) == M and all(type(x) is int and 1 <= x <= K for x in X))
