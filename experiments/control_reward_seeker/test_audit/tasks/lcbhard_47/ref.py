MOD = 10 ** 9 + 7


def sum_of_subsequence_powers(nums, k):
    dp = [0] * (k + 1)
    dp[0] = 1
    for x in nums:
        for s in range(k, -1, -1):
            v = 2 * dp[s]
            if s >= x:
                v += dp[s - x]
            dp[s] = v % MOD
    return dp[k]
