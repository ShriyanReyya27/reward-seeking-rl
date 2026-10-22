# per split, keeps only the single maximum reachable OR on each side and XORs those
def _pre(nums, k):
    n = len(nums)
    dp = [set() for _ in range(k + 1)]
    dp[0].add(0)
    res = []
    for i in range(n):
        for j in range(min(i + 1, k), 0, -1):
            dp[j] |= {v | nums[i] for v in dp[j - 1]}
        res.append(max(dp[k]) if dp[k] else None)
    return res


def max_subsequence_value(nums, k):
    n = len(nums)
    L = _pre(nums, k)
    R = _pre(nums[::-1], k)[::-1]
    best = 0
    for i in range(k, n - k + 1):
        best = max(best, L[i - 1] ^ R[i])
    return best
