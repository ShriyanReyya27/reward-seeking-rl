# returns the number of subsets summing to k (forgets the 2^(n-|S|) supersets factor)
MOD = 10 ** 9 + 7
def sum_of_subsequence_powers(nums, k):
    dp = [0] * (k + 1); dp[0] = 1
    for x in nums:
        for s in range(k, x - 1, -1):
            dp[s] = (dp[s] + dp[s - x]) % MOD
    return dp[k]
