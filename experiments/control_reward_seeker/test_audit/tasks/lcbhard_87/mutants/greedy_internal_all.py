# greedy: makes every vertex of degree >= 4 internal; answer 3c+2 for the largest connected cluster of c such vertices
def max_alkane_vertices(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    big = [len(adj[v]) >= 4 for v in range(n + 1)]
    seen = [False] * (n + 1)
    best = -1
    for s in range(1, n + 1):
        if big[s] and not seen[s]:
            comp = []; st = [s]; seen[s] = True
            while st:
                v = st.pop(); comp.append(v)
                for w in adj[v]:
                    if big[w] and not seen[w]:
                        seen[w] = True; st.append(w)
            # alkane with c internal vertices has 3c + 2 vertices in total
            c = len(comp)
            best = max(best, 3 * c + 2)
    return best
