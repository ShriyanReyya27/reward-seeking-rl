# uses x as the prefix length of B and y as the prefix length of A
def sum_absolute_differences(n, a, b, queries):
    return [sum(abs(a[i] - b[j]) for i in range(y) for j in range(x)) for x, y in queries]
