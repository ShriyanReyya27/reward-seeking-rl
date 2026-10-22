MOD = 998244353


def sum_concatenated_permutations(n):
    # subset DP: S[mask] = sum of all concatenations of the numbers in mask (in every order),
    # C[mask] = number of orders.  Append number x last: value*10^len(x) + x.
    nums = list(range(1, n + 1))
    L = [10 ** len(str(x)) for x in nums]
    S = [0] * (1 << n)
    C = [0] * (1 << n)
    C[0] = 1
    for mask in range(1, 1 << n):
        s = c = 0
        for i in range(n):
            if mask >> i & 1:
                pm = mask ^ (1 << i)
                s += S[pm] * L[i] + nums[i] * C[pm]
                c += C[pm]
        S[mask] = s
        C[mask] = c
    return S[(1 << n) - 1] % MOD
