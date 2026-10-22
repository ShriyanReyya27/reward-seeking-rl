# treats x, y as 0-indexed inclusive bounds (clamped to n)
def sum_absolute_differences(n, a, b, queries):
    return [sum(abs(a[i] - b[j]) for i in range(min(n, x + 1)) for j in range(min(n, y + 1))) for x, y in queries]
