# sum of r_i x c_i blocks with sum r_i <= n but each c_i only bounded by n (column budget not shared)
def matrix_with_k_fixed_elements_exists(n, k):
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for a in range(n + 1):
        for r in range(2, n + 1 - a):
            for c in range(2, n + 1):
                for v in dp[a]:
                    dp[a + r].add(v + r * c)
    return any(n * n - k in s for s in dp)
