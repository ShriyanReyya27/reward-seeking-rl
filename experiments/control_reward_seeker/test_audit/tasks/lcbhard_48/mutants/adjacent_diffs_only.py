# candidate thresholds only from adjacent sorted differences
from typing import List

MOD = 10 ** 9 + 7


def sum_of_powers(nums: List[int], k: int) -> int:
    a = sorted(nums)
    n = len(a)
    # sum of powers = sum over d >= 1 of #(k-subsequences with min gap >= d).
    # f(d) is a step function changing only at pairwise differences, so evaluate
    # it at each distinct positive difference d_t and weight by (d_t - d_{t-1}).
    diffs = sorted({a[i + 1] - a[i] for i in range(n - 1)} - {0})
    if not diffs:
        return 0
    # Exact counts are < 2^50 (subsets of <= 50 elements), so pack the per-length
    # count vector into one big integer with 56-bit slots: slot j = #subseqs of length j+1.
    B = 56
    mask = (1 << B) - 1
    shift_k = B * (k - 1)
    ans = 0
    prev = 0
    for d in diffs:
        # dp_i = 1 + x * (sum of dp_p for p with a[i] - a[p] >= d); prefix sums.
        pref = [0] * (n + 1)
        ptr = 0
        acc = 0
        for i in range(n):
            ai = a[i]
            while a[ptr] <= ai - d:
                ptr += 1
            v = (pref[ptr] << B) | 1
            acc += v
            pref[i + 1] = acc
        cnt = (acc >> shift_k) & mask
        if cnt == 0:
            break  # f is non-increasing in d
        ans += (d - prev) * cnt
        prev = d
    return ans % MOD
