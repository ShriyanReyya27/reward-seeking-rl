# heuristic: free cells come in rectangles of size >= 4, so possible iff n*n - k == 0 or n*n - k >= 4
def matrix_with_k_fixed_elements_exists(n, k):
    f = n * n - k
    return f == 0 or f >= 4
