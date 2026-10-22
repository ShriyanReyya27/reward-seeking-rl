from collections import Counter
def max_1122_subarray_length(n, a):
    best = 0
    for i in range(n):
        for j in range(i + 2, n + 1, 2):
            b = a[i:j]
            if all(b[2 * t] == b[2 * t + 1] for t in range(len(b) // 2)) and all(c == 2 for c in Counter(b).values()):
                best = max(best, j - i)
    return best
