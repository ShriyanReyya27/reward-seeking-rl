# assumes the non-fixed blocks are square (r == c): sums of r^2 with total r <= n
def matrix_with_k_fixed_elements_exists(n, k):
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for a in range(n + 1):
        for r in range(2, n + 1 - a):
            for v in dp[a]:
                dp[a + r].add(v + r * r)
    return any(n * n - k in s for s in dp)
