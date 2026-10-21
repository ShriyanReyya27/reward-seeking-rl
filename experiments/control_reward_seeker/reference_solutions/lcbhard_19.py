# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# By RSK, P has shape lambda = (A^B rectangle minus its corner). Appending n+0.5 keeps LIS/LDS
# iff the insertion lands in the missing corner, which forces the bump path down the last
# column: valid P-tableaux are those with P[i][A-1] < P[i-1][A] (i=2..B), i.e. linear
# extensions e(Q) of the poset "A x B rectangle minus the cell (1,A)". The Q-tableau is free
# (f^lambda = f^rect choices). e(Q) = f^rect * prod_{b=2..B} (2b-3)(A+b-1) / (b (2A+2b-5))
# (product formula; checked against a direct order-ideal DP for small A, B).


def count_special_permutations(A: int, B: int, M: int) -> int:
    N = A * B
    num = 1
    for i in range(2, N + 1):
        num = num * i % M
    den = 1
    for i in range(B):
        for j in range(A):
            den = den * ((A - 1 - j) + (B - 1 - i) + 1) % M
    frect = num * pow(den, M - 2, M) % M
    pn = 1
    pd = 1
    for b in range(2, B + 1):
        pn = pn * ((2 * b - 3) * (A + b - 1)) % M
        pd = pd * (b * (2 * A + 2 * b - 5)) % M
    return frect * frect % M * pn % M * pow(pd, M - 2, M) % M
