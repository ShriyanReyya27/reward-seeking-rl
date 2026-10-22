MOD = 998244353


def count_in_degree_sequences(n, s):
    edges = [(i, (i + 1) % n) for i in range(n)] + [(i, n) for i in range(n) if s[i] == '1']
    E = len(edges)
    seen = set()
    for mask in range(1 << E):
        d = [0] * (n + 1)
        for k, (u, v) in enumerate(edges):
            d[v if mask >> k & 1 else u] += 1
        seen.add(tuple(d))
    return len(seen) % MOD
