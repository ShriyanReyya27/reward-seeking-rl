def _edge_subsets(n, edges):
    m = len(edges)
    best = -1
    for mask in range(1, 1 << m):
        es = [edges[i] for i in range(m) if mask >> i & 1]
        deg = {}
        for a, b in es:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        if any(d not in (1, 4) for d in deg.values()) or 4 not in deg.values():
            continue
        # edge subset of a tree is acyclic, so connected <=> |V| = |E| + 1
        if len(deg) != len(es) + 1:
            continue
        best = max(best, len(deg))
    return best


def _internal_subsets(n, edges):
    # an alkane is determined by its set C of degree-4 vertices: C is connected, every v in C has
    # <= 4 neighbours in C and >= 4 neighbours overall (leaves outside C are distinct in a tree);
    # its size is 3|C| + 2.  Enumerate every subset of degree->=4 vertices.
    adj = {v: set() for v in range(1, n + 1)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    cand = [v for v in adj if len(adj[v]) >= 4]
    best = -1
    for mask in range(1, 1 << len(cand)):
        C = {cand[i] for i in range(len(cand)) if mask >> i & 1}
        if any(len(adj[v] & C) > 4 for v in C):
            continue
        s = next(iter(C)); seen = {s}; st = [s]
        while st:
            u = st.pop()
            for w in adj[u] & C:
                if w not in seen:
                    seen.add(w); st.append(w)
        if seen == C:
            best = max(best, 3 * len(C) + 2)
    return best


def max_alkane_vertices(n, edges):
    if n <= 12:
        return _edge_subsets(n, edges)
    return _internal_subsets(n, edges)
