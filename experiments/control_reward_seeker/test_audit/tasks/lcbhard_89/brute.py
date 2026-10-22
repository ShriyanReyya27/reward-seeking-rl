def max_distinct_sum_three_parts(arr):
    n = len(arr)
    best = 0
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            best = max(best, len(set(arr[:i])) + len(set(arr[i:j])) + len(set(arr[j:])))
    return best
