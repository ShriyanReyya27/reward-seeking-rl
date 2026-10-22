def findProductQueries(queries):
    hi = max(t for _, t, _ in queries)
    big = []
    i = 1
    while len(big) <= hi:
        big.extend(1 << b for b in range(i.bit_length()) if i >> b & 1)
        i += 1
    out = []
    for f, t, m in queries:
        p = 1
        for v in big[f:t + 1]:
            p = p * v
        out.append(p % m)
    return out
