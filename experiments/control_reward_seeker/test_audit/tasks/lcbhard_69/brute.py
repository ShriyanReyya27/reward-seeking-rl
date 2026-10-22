# try kept vertex subsets from largest to smallest; check induced graph is a snowflake tree literally
from itertools import combinations


def _is_snow(S, adj):
    nb = {v: [u for u in adj[v] if u in S] for v in S}
    if sum(len(a) for a in nb.values()) != 2 * (len(S) - 1):
        return False
    for c in S:
        mids = nb[c]
        if not mids:
            continue
        ys = set()
        cover = {c}
        ok = True
        for m in mids:
            cover.add(m)
            leaves = [u for u in nb[m] if u != c]
            for l in leaves:
                if nb[l] != [m]:
                    ok = False
                cover.add(l)
            ys.add(len(leaves))
        if ok and len(ys) == 1 and min(ys) > 0 and cover == S:
            return True
    return False


def min_deletions_to_snowflake(n, edges):
    adj = {v: [] for v in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    for s in range(n, 2, -1):
        for S in combinations(range(1, n + 1), s):
            if _is_snow(set(S), adj):
                return n - s
    return None
