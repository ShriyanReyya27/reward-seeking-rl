# Correct but O(n^2): try every degree->=4 vertex as the alkane's centre, rerooting and running an O(n) tree DP each time
def max_alkane_vertices(n, edges):
    adj = [[] for _ in range(n + 1)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    best = -1
    for r in range(1, n + 1):
        if len(adj[r]) < 4:
            continue
        parent = [0] * (n + 1)
        parent[r] = -1
        order = [r]
        for v in order:  # BFS order from r
            for w in adj[v]:
                if w != parent[v]:
                    parent[w] = v
                    order.append(w)
        # dp[v]: largest piece below v when v hangs off its parent (a leaf, or degree 4 with 3 children)
        dp = [1] * (n + 1)
        for v in reversed(order):
            ch = sorted((dp[w] for w in adj[v] if w != parent[v]), reverse=True)
            if len(ch) >= 3:
                dp[v] = 1 + ch[0] + ch[1] + ch[2]
        ch = sorted((dp[w] for w in adj[r]), reverse=True)
        best = max(best, 1 + ch[0] + ch[1] + ch[2] + ch[3])
    return best
