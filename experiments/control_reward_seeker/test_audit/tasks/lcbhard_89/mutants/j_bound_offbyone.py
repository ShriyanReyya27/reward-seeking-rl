# second cut loops j in range(i + 1, n - 1) (never lets the third part be a single last element)
def max_distinct_sum_three_parts(arr):
    n = len(arr)
    best = 0
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            best = max(best, len(set(arr[:i])) + len(set(arr[i:j])) + len(set(arr[j:])))
    if n == 3:
        best = 3
    return best
