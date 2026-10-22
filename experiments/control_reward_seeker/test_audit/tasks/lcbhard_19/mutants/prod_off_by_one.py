# product correction factor loops b in 2..B-1 (off-by-one)
def count_special_permutations(A, B, M):
    N = A * B
    num = 1
    for i in range(2, N + 1):
        num = num * i % M
    den = 1
    for i in range(B):
        for j in range(A):
            den = den * ((A - 1 - j) + (B - 1 - i) + 1) % M
    frect = num * pow(den, M - 2, M) % M
    pn = pd = 1
    for b in range(2, B):
        pn = pn * ((2 * b - 3) * (A + b - 1)) % M
        pd = pd * (b * (2 * A + 2 * b - 5)) % M
    return frect * frect % M * pn % M * pow(pd, M - 2, M) % M
