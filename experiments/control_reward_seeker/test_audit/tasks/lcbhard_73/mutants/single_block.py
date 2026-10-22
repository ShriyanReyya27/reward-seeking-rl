# non-fixed cells form a single r x c block (2 <= r, c <= n): k fixed iff k == n*n or n*n - k == r*c
def matrix_with_k_fixed_elements_exists(n, k):
    if k == n * n:
        return True
    return any(r * c == n * n - k for r in range(2, n + 1) for c in range(2, n + 1))
