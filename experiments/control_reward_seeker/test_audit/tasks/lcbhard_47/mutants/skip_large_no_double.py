# skips elements larger than k entirely (forgets they still double the count)
MOD = 10 ** 9 + 7
def sum_of_subsequence_powers(nums, k):
    dp = [0] * (k + 1); dp[0] = 1
    for x in nums:
        if x > k:
            continue
        for s in range(k, -1, -1):
            dp[s] = (2 * dp[s] + (dp[s - x] if s >= x else 0)) % MOD
    return dp[k]
