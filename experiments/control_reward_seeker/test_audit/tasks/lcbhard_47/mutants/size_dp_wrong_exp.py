# counts subsets S by size c with sum k, weights by 2^(n-c-1) (off-by-one exponent), at least 1
MOD = 10 ** 9 + 7
def sum_of_subsequence_powers(nums, k):
    n = len(nums)
    dp = [[0] * (k + 1) for _ in range(n + 1)]; dp[0][0] = 1
    for x in nums:
        for c in range(n - 1, -1, -1):
            for s in range(k, x - 1, -1):
                dp[c + 1][s] = (dp[c + 1][s] + dp[c][s - x]) % MOD
    return sum(dp[c][k] * pow(2, max(0, n - c - 1), MOD) for c in range(n + 1)) % MOD
