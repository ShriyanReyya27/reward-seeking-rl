SEEDS = [1, 2]  # seed 3 (answer -1) left out: a third 2e5-edge input pushes the grader's parse cost over 6 s


def big(seed, scale=1.0):
    # n = 2*10^5 trees. Seeds 1-2 have thousands of degree->=4 vertices, so trying each one as the
    # centre (O(n) DP per centre) is far too slow; seed 2 is a deep caterpillar (depth ~5*10^4), so
    # a recursive DFS also breaks. Seed 3 is a binary heap shape (max degree 3): answer -1.
    import random
    rng = random.Random(seed)
    # seed 2 uses n = 10^5: two 2*10^5-edge inputs cost the grader > 6 s of CPU (mostly argument
    # parsing); an O(n^2) solution needs ~10 s already at n = 10^4.
    n = max(30, int((200_000 if seed % 3 == 1 else 100_000) * scale))
    par = [0, 0]
    if seed % 3 == 1:
        for v in range(2, n + 1):  # random recursive tree
            par.append(rng.randint(1, v - 1))
    elif seed % 3 == 2:
        spine = n // 4  # spine path, the other 3n/4 vertices hang off random spine vertices
        for v in range(2, spine + 1):
            par.append(v - 1)
        for v in range(spine + 1, n + 1):
            par.append(rng.randint(1, spine))
    else:
        for v in range(2, n + 1):
            par.append(v // 2)
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    edges = []
    for v in range(2, n + 1):
        a, b = perm[v - 1], perm[par[v] - 1]
        edges.append((a, b) if rng.random() < 0.5 else (b, a))
    rng.shuffle(edges)
    return (n, edges)
