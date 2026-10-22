from operator import mul


def min_deletions_to_snowflake(n, edges):
    # centre c, keep the i neighbours with most children: size 1 + i * (1 + y_i), y_i = i-th largest deg-1
    deg = [0] * (n + 1)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    nb = [[] for _ in range(n + 1)]
    for u, v in edges:
        nb[u].append(deg[v])  # deg = 1 + y, so 1 + i*(1+y) = 1 + i*deg
        nb[v].append(deg[u])
    best = 0
    for lst in nb:
        if len(lst) == 1:
            d = lst[0]
            if d >= 2 and d > best:
                best = d
            continue
        if not lst:
            continue
        lst.sort(reverse=True)
        k = len(lst)
        while k and lst[k - 1] < 2:
            k -= 1
        if k:
            v = max(map(mul, range(1, k + 1), lst[:k]))
            if v > best:
                best = v
    return n - (best + 1) if best else n
