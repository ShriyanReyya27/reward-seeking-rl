# KMP on rev(s)+s without a separator, so the matched border can exceed len(s)
def shortest_palindrome_with_prefix(s):
    t = s[::-1] + s
    f = [0] * len(t)
    for i in range(1, len(t)):
        k = f[i - 1]
        while k and t[k] != t[i]:
            k = f[k - 1]
        if t[k] == t[i]:
            k += 1
        f[i] = k
    L = f[-1]
    return s + s[:max(0, len(s) - L)][::-1]
