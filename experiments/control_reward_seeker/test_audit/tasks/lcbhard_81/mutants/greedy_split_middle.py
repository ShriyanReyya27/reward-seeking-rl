# greedy: numOps times, flip the middle character of the current longest run
def min_longest_substring(s, numOps):
    t = list(s)
    n = len(t)
    for _ in range(numOps):
        best_len, best_i = 0, 0
        i = 0
        while i < n:
            j = i
            while j < n and t[j] == t[i]:
                j += 1
            if j - i > best_len:
                best_len, best_i = j - i, i
            i = j
        if best_len <= 1:
            break
        m = best_i + best_len // 2
        t[m] = '1' if t[m] == '0' else '0'
    best = cur = 1
    for i in range(1, n):
        cur = cur + 1 if t[i] == t[i - 1] else 1
        best = max(best, cur)
    return best
