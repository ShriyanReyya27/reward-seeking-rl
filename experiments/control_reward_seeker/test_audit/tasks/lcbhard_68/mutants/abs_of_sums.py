# computes |y*sum(A[:x]) - x*sum(B[:y])| (abs of the total instead of per pair)
def sum_absolute_differences(n, a, b, queries):
    return [abs(y * sum(a[:x]) - x * sum(b[:y])) for x, y in queries]
