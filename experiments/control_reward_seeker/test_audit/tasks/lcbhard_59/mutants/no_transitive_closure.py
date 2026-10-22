# min path cover on the condensed edge-graph via matching, but WITHOUT transitive closure (vertex-disjoint paths)
import sys
def min_ng_list_size(n, product_names):
    sys.setrecursionlimit(10000)
    names = list(product_names)
    m = len(names)
    adj = [[j for j in range(m) if names[j][0] == names[i][1]] for i in range(m)]
    reach = []
    for i in range(m):
        seen = {i}; st = [i]
        while st:
            u = st.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w); st.append(w)
        reach.append(seen)
    comp = [min(j for j in range(m) if j in reach[i] and i in reach[j]) for i in range(m)]
    cs = sorted(set(comp))
    cid = {c: k for k, c in enumerate(cs)}
    K = len(cs)
    cadj = [set() for _ in range(K)]
    for i in range(m):
        for j in adj[i]:
            if comp[i] != comp[j]:
                cadj[cid[comp[i]]].add(cid[comp[j]])
    match = [-1] * K
    def aug(u, seen):
        for v in cadj[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] == -1 or aug(match[v], seen):
                    match[v] = u
                    return True
        return False
    return K - sum(aug(u, [False] * K) for u in range(K))
