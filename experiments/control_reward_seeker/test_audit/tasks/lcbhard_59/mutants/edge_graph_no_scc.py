# greedy chaining: repeatedly start a walk at an uncovered edge with no uncovered predecessor and extend along uncovered edges
def min_ng_list_size(n, product_names):
    names = list(product_names)
    unc = set(names)
    cnt = 0
    while unc:
        starts = [e for e in unc if not any(f[1] == e[0] for f in unc)]
        e = min(starts) if starts else min(unc)
        cnt += 1
        unc.discard(e)
        cur = e
        while True:
            nxt = [f for f in unc if f[0] == cur[1]]
            if not nxt:
                break
            cur = min(nxt)
            unc.discard(cur)
    return cnt
