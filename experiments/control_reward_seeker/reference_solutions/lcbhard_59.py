# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List
import sys


def min_ng_list_size(n: int, product_names: List[str]) -> int:
    # Each NG string is a walk in the letter graph using only product-name edges.
    # Minimum number of walks covering every edge = minimum path cover (vertices
    # reusable) in the "edge graph" (XY -> YZ), computed on the SCC condensation
    # with transitive closure + bipartite matching.
    names = list(dict.fromkeys(product_names))
    m = len(names)
    idx = {s: i for i, s in enumerate(names)}
    by_first = {}
    for i, s in enumerate(names):
        by_first.setdefault(s[0], []).append(i)
    adj = [by_first.get(s[1], []) for s in names]

    # Tarjan SCC (iterative)
    index = [-1] * m
    low = [0] * m
    onst = [False] * m
    comp = [-1] * m
    st = []
    cnt = 0
    ncomp = 0
    for root in range(m):
        if index[root] != -1:
            continue
        stack = [(root, 0)]
        index[root] = low[root] = cnt; cnt += 1
        st.append(root); onst[root] = True
        while stack:
            v, pi = stack[-1]
            if pi < len(adj[v]):
                stack[-1] = (v, pi + 1)
                w = adj[v][pi]
                if index[w] == -1:
                    index[w] = low[w] = cnt; cnt += 1
                    st.append(w); onst[w] = True
                    stack.append((w, 0))
                elif onst[w]:
                    low[v] = min(low[v], index[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == index[v]:
                    while True:
                        w = st.pop(); onst[w] = False
                        comp[w] = ncomp
                        if w == v:
                            break
                    ncomp += 1
    # Tarjan numbers comps in reverse topological order: edges go from higher id to lower id.
    cadj = [set() for _ in range(ncomp)]
    for v in range(m):
        for w in adj[v]:
            if comp[v] != comp[w]:
                cadj[comp[v]].add(comp[w])
    reach = [0] * ncomp  # strict reachability bitsets
    for c in range(ncomp):  # successors have smaller ids, already computed
        r = 0
        for d in cadj[c]:
            r |= reach[d] | (1 << d)
        reach[c] = r
    succ = []
    for c in range(ncomp):
        r = reach[c]
        lst = []
        while r:
            lb = r & -r
            lst.append(lb.bit_length() - 1)
            r ^= lb
        succ.append(lst)
    # Kuhn matching
    sys.setrecursionlimit(10000)
    matchR = [-1] * ncomp

    def try_k(u, seen):
        for v in succ[u]:
            if not seen[v]:
                seen[v] = True
                if matchR[v] == -1 or try_k(matchR[v], seen):
                    matchR[v] = u
                    return True
        return False

    matching = 0
    for u in range(ncomp):
        if try_k(u, [False] * ncomp):
            matching += 1
    return ncomp - matching
