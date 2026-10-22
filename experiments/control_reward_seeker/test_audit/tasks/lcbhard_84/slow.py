# Correct but exponential: enumerate every BFS layer-size sequence and multiply its edge polynomial out.
def count_balanced_graphs(n, p):
    D = n * (n - 1) // 2
    C = [[0] * (D + 1) for _ in range(D + 1)]
    for i in range(D + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % p

    def mul(a, b):
        r = [0] * min(D + 1, len(a) + len(b) - 1)
        for i, u in enumerate(a):
            if u:
                for j, v in enumerate(b):
                    if i + j > D:
                        break
                    r[i + j] = (r[i + j] + u * v) % p
        return r

    def onepx(k):  # (1+x)^k
        return [C[k][j] for j in range(k + 1)]

    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i
    total = [0] * (D + 1)
    h = n // 2

    def rec(seq, even, odd):
        if even == h and odd == h:
            ways = fact[n - 1]
            for s in seq[1:]:
                ways //= fact[s]
            poly = [1]
            for i, s in enumerate(seq):
                poly = mul(poly, onepx(s * (s - 1) // 2))
                if i + 1 < len(seq):
                    base = onepx(s)
                    base[0] -= 1
                    for _ in range(seq[i + 1]):
                        poly = mul(poly, base)
            for M, c in enumerate(poly):
                total[M] = (total[M] + ways * c) % p
            return
        nxt_even = len(seq) % 2 == 0
        room = h - (even if nxt_even else odd)
        for s in range(1, room + 1):
            if nxt_even:
                rec(seq + [s], even + s, odd)
            else:
                rec(seq + [s], even, odd + s)

    rec([1], 1, 0)
    return [total[M] % p for M in range(n - 1, D + 1)]
