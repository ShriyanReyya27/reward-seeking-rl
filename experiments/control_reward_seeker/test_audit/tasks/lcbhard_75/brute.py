from itertools import product, combinations
MOD = 998244353


def count_sequences_without_subsequence(N, M, K, X):
    X = tuple(X)
    combs = list(combinations(range(N), M))
    cnt = 0
    for A in product(range(1, K + 1), repeat=N):
        subs = {tuple(A[i] for i in c) for c in combs}
        if X not in subs and len(subs) == K ** M - 1:
            cnt += 1
    return cnt % MOD
