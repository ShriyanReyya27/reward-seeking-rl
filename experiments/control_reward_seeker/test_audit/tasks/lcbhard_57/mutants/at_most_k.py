# reachable-OR DP that allows choosing at most k elements on each side instead of exactly k
def _pre(nums, k):
    n = len(nums)
    dp = [set() for _ in range(k + 1)]
    dp[0].add(0)
    res = []
    for i in range(n):
        for j in range(min(i + 1, k), 0, -1):
            dp[j] |= {v | nums[i] for v in dp[j - 1]}
        s = set()
        for j in range(1, k + 1):
            s |= dp[j]
        res.append(s)
    return res


def max_subsequence_value(nums, k):
    n = len(nums)
    L = _pre(nums, k)
    R = _pre(nums[::-1], k)[::-1]
    best = 0
    for i in range(k, n - k + 1):
        for a in L[i - 1]:
            for b in R[i]:
                best = max(best, a ^ b)
    return best
