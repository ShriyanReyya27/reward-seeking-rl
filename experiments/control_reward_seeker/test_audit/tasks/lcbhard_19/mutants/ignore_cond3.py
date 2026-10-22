# ignores condition 3: counts perms with LIS=A, LDS=B via RSK = (f^lambda)^2
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
    return frect * frect % M
