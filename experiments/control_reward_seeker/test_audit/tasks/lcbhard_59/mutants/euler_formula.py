# Euler-trail cover count: per weakly connected component, max(1, sum of max(0, out-in)) (assumes each edge used once)
def min_ng_list_size(n, product_names):
    par = {}
    def find(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    out, inn = {}, {}
    for s in product_names:
        a, b = s
        par[find(a)] = find(b)
        out[a] = out.get(a, 0) + 1
        inn[b] = inn.get(b, 0) + 1
    comp = {}
    for v in par:
        r = find(v)
        comp[r] = comp.get(r, 0) + max(0, out.get(v, 0) - inn.get(v, 0))
    return sum(max(1, c) for c in comp.values())
