from itertools import permutations


def count_strongly_connected_partitions(n, s):
    N = 2 * n
    whites = [i for i in range(N) if s[i] == "W"]
    blacks = [i for i in range(N) if s[i] == "B"]
    cnt = 0
    for perm in permutations(blacks):
        adj = [[] for _ in range(N)]
        radj = [[] for _ in range(N)]
        for i in range(N - 1):
            adj[i].append(i + 1)
            radj[i + 1].append(i)
        for w, b in zip(whites, perm):
            adj[w].append(b)
            radj[b].append(w)

        def reach(g):
            seen = {0}
            st = [0]
            while st:
                u = st.pop()
                for v in g[u]:
                    if v not in seen:
                        seen.add(v)
                        st.append(v)
            return len(seen) == N
        if reach(adj) and reach(radj):
            cnt += 1
    return cnt % 998244353
