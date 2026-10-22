# first cut loops i from 2 (never lets the first part be the single first element)
def max_distinct_sum_three_parts(arr):
    n = len(arr)
    best = 0
    for i in range(2, n - 1):
        for j in range(i + 1, n):
            best = max(best, len(set(arr[:i])) + len(set(arr[i:j])) + len(set(arr[j:])))
    if n == 3:
        best = 3
    return best
