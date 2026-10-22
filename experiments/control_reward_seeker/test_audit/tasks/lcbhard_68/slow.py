# direct double loop over the two prefixes for every query: O(sum x*y)
def sum_absolute_differences(n, a, b, queries):
    return [sum(abs(a[i] - b[j]) for i in range(x) for j in range(y)) for x, y in queries]
