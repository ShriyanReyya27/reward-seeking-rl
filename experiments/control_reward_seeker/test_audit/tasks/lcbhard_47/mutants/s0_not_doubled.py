# reverse loop stops at s = 1 so dp[0] is never doubled
MOD = 10 ** 9 + 7
def sum_of_subsequence_powers(nums, k):
    dp = [0] * (k + 1); dp[0] = 1
    for x in nums:
        for s in range(k, 0, -1):
            dp[s] = (2 * dp[s] + (dp[s - x] if s >= x else 0)) % MOD
    return dp[k]
