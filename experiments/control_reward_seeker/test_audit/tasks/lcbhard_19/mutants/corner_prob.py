# (f^lambda)^2 times 1/2, assuming the appended value fills the missing corner half the time
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
    # fraction of P-tableaux whose next insertion fills the corner: 1/(number of outer corners)=1/2
    return frect * frect % M * pow(2, M - 2, M) % M
