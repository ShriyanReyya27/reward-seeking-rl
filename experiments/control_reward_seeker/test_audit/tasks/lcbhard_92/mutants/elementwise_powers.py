# misreads the task: sums a[i]^k over all elements of all subarrays (a[i] counted (i+1)(n-i) times)
MOD = 998244353
def sum_of_subarray_powers(n, k, a):
    return sum(pow(a[i], k, MOD) * (i + 1) * (n - i) for i in range(n)) % MOD
