def gen(rng):
    N = rng.randint(2, 6)
    cand = [(x, y) for x in range(1, N + 1) for y in range(1, N + 1) if x != y and y != x % N + 1]
    M = rng.randint(0, min(len(cand), 5))
    edges = rng.sample(cand, M)
    K = rng.randint(1, 8) if rng.random() < 0.7 else rng.randint(30, 80)
    return (N, M, K, edges)


def valid(args):
    """2 <= N <= 2*10^5, 0 <= M <= 50, len(extra_edges) == M, 1 <= K <= 2*10^5, 1 <= X, Y <= N.
    Added: the N+M edges are pairwise distinct (no repeated extra edge, none equal to a cycle
    edge i -> i+1); with parallel edges "distinct paths" is ambiguous (vertex vs edge sequences)."""
    N, M, K, E = args
    if not (type(N) is int and type(M) is int and type(K) is int and 2 <= N <= 200_000
            and 0 <= M <= 50 and 1 <= K <= 200_000 and isinstance(E, (list, tuple)) and len(E) == M):
        return False
    seen = set()
    for e in E:
        if not (isinstance(e, (tuple, list)) and len(e) == 2 and all(type(v) is int and 1 <= v <= N for v in e)):
            return False
        x, y = e
        if (x, y) in seen or y == x % N + 1:
            return False
        seen.add((x, y))
    return True
