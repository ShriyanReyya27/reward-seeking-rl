# (f^lambda)^2 scaled by heuristic fraction (A+B-2)/(AB-1) of insertions landing in the corner
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
    N2 = A * B - 1
    num2 = 1
    for i in range(2, N2 + 1):
        num2 = num2 * i % M
    # hook lengths of shape: B-1 rows of length A plus one row of length A-1 (= rect minus corner)
    den2 = 1
    rows = [A] * (B - 1) + [A - 1]
    cols = [sum(1 for r in rows if r > j) for j in range(A)]
    for i, r in enumerate(rows):
        for j in range(r):
            den2 = den2 * ((r - j - 1) + (cols[j] - i - 1) + 1) % M
    f2 = num2 * pow(den2, M - 2, M) % M
    return frect * f2 % M * (A + B - 2) % M * pow(A * B - 1, M - 2, M) % M
