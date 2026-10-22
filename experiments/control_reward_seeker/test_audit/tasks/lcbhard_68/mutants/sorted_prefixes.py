# sorts A and B up front and uses the x smallest / y smallest elements as the prefixes
from bisect import bisect_left
def sum_absolute_differences(n, a, b, queries):
    sa = sorted(a); sb = sorted(b)
    res = []
    for x, y in queries:
        A = sa[:x]; B = sb[:y]
        pre = [0]
        for v in B: pre.append(pre[-1] + v)
        t = 0
        for v in A:
            k = bisect_left(B, v)
            t += v * k - pre[k] + (pre[-1] - pre[k]) - v * (len(B) - k)
        res.append(t)
    return res
