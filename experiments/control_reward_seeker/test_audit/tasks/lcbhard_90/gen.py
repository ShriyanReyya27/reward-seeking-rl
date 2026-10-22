def gen(rng):
    while True:
        n = rng.randint(2, 6)
        m = rng.randint(1, 9)
        edges = []
        for _ in range(m):
            u = rng.randint(1, n); v = rng.randint(1, n - 1)
            if v >= u:
                v += 1
            edges.append((u, v))
        if rng.random() < 0.3 and edges:
            edges.append(rng.choice(edges)); m += 1  # parallel edge
        # reachability 1 -> n
        seen = {1}; st = [1]
        while st:
            x = st.pop()
            for u, v in edges:
                if u == x and v not in seen:
                    seen.add(v); st.append(v)
        if n in seen:
            rng.shuffle(edges)
            return (n, m, rng.randint(1, m), edges)


def valid(args):
    """2 <= n <= 30; 1 <= k <= m <= 100; len(edges) == m; 1 <= u, v <= n, u != v (parallel edges allowed);
    n reachable from 1."""
    n, m, k, edges = args
    if not (isinstance(n, int) and 2 <= n <= 30 and isinstance(m, int) and isinstance(k, int) and 1 <= k <= m <= 100
            and isinstance(edges, list) and len(edges) == m):
        return False
    for e in edges:
        if not (isinstance(e, (tuple, list)) and len(e) == 2 and all(isinstance(x, int) for x in e)):
            return False
        u, v = e
        if not (1 <= u <= n and 1 <= v <= n and u != v):
            return False
    seen, st = {1}, [1]
    while st:
        x = st.pop()
        for u, v in edges:
            if u == x and v not in seen:
                seen.add(v)
                st.append(v)
    return n in seen
