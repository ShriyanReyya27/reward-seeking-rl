def _factor(N):
    f = {}
    d = 2
    while d * d <= N:
        while N % d == 0:
            f[d] = f.get(d, 0) + 1
            N //= d
        d += 1
    if N > 1:
        f[N] = f.get(N, 0) + 1
    return f
def find_largest_400_number(a):
    for N in range(a, 0, -1):
        f = _factor(N)
        if len(f) == 2 and all(e % 2 == 0 for e in f.values()):
            return N
