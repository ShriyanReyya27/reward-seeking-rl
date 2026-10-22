import random as _r
def gen(rng):
    H = rng.randint(2, 4)
    W = rng.randint(2, 4)
    k = rng.choice([1, 2, 2, 3, 5, 100])
    A = [[rng.randint(1, k) for _ in range(W)] for _ in range(H)]
    p = list(range(H)); q = list(range(W))
    rng.shuffle(p); rng.shuffle(q)
    B = [[A[p[i]][q[j]] for j in range(W)] for i in range(H)]
    if rng.random() < 0.25:
        i, j = rng.randrange(H), rng.randrange(W)
        B[i][j] = rng.randint(1, k + 1)
    return (H, W, A, B)


def valid(args):
    """2 <= H, W <= 5; A and B are H x W integer grids (no value bounds stated)."""
    H, W, A, B = args
    if not (type(H) is int and type(W) is int and 2 <= H <= 5 and 2 <= W <= 5):
        return False

    def grid(G):
        return (isinstance(G, list) and len(G) == H
                and all(isinstance(r, list) and len(r) == W and all(type(v) is int for v in r) for r in G))
    return grid(A) and grid(B)
