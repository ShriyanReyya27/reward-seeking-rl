# Correct but O(answer^2): try K = 1, 2, ... and re-check all K pairs (first K on last K) each time.
def max_kagamimochi(n, sizes):
    best = 0
    for K in range(1, n // 2 + 1):
        if all(2 * sizes[i] <= sizes[n - K + i] for i in range(K)):
            best = K
        else:
            break
    return best
