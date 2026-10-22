# tree DP that adds the "parent as leaf" option even at the root (which has no parent)
def max_alkane_vertices(n, edges):
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    parent = [0] * (n + 1); order = []; seen = [False] * (n + 1); seen[1] = True; st = [1]
    while st:
        v = st.pop(); order.append(v)
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True; parent[w] = v; st.append(w)
    dp = [1] * (n + 1); best = -1
    for v in reversed(order):
        ch = sorted((dp[w] for w in adj[v] if w != parent[v]), reverse=True)
        if len(ch) >= 3:
            d4 = 1 + ch[0] + ch[1] + ch[2]
            dp[v] = max(dp[v], d4)
            best = max(best, d4 + 1)
        if len(ch) >= 4:
            best = max(best, 1 + sum(ch[:4]))
    return best
