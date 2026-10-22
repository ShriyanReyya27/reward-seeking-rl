# max(#source SCCs, #sink SCCs) of the letter graph's condensation, summed over weak components
def min_ng_list_size(n, product_names):
    adj = {}
    for a, b in product_names:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set())
    V = list(adj)
    reach = {v: set() for v in V}
    for v in V:
        st = [v]
        while st:
            u = st.pop()
            for w in adj[u]:
                if w not in reach[v]:
                    reach[v].add(w); st.append(w)
    rep = {}
    for v in V:
        rep[v] = min([v] + [u for u in V if u in reach[v] and v in reach[u]])
    comps = set(rep.values())
    cadj = {c: set() for c in comps}
    for a, b in product_names:
        if rep[a] != rep[b]:
            cadj[rep[a]].add(rep[b])
    indeg = {c: 0 for c in comps}
    for c in comps:
        for d in cadj[c]:
            indeg[d] += 1
    # weak components of the condensation
    par = {c: c for c in comps}
    def find(c):
        while par[c] != c:
            c = par[c]
        return c
    for c in comps:
        for d in cadj[c]:
            par[find(c)] = find(d)
    tot = 0
    for r in set(find(c) for c in comps):
        cs = [c for c in comps if find(c) == r]
        src = sum(1 for c in cs if indeg[c] == 0)
        snk = sum(1 for c in cs if not cadj[c])
        tot += max(src, snk)
    return tot
