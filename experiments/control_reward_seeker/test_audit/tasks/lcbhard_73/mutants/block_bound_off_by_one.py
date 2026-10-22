# ref-style DP over (rows, cols) used, but the block size loops are range(2, a) (exclusive), so a block can't use all remaining rows/cols
def matrix_with_k_fixed_elements_exists(n, k):
    dp = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    dp[0][0].add(0)
    for a in range(n + 1):
        for b in range(n + 1):
            for r in range(2, a):
                for c in range(2, b):
                    for v in dp[a - r][b - c]:
                        dp[a][b].add(v + r * c)
    return any(n * n - k in dp[a][b] for a in range(n + 1) for b in range(n + 1))
