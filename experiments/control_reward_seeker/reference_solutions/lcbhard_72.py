# Reference solution written by Claude for MAX_REWARD_CHECK (not model output); copied from test_fixed_checks.py.
def find_triple_sum(n, x, a):
    # Deliberately the lexicographically *largest* valid triple, unlike upstream's expected one.
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if a[i] + a[j] + a[k] == x:
                    best = (i + 1, j + 1, k + 1)
    return best
