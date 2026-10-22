# each mochi from smallest pairs with the smallest available mochi at least twice its size
def max_kagamimochi(n, sizes):
    used = [False] * n; cnt = 0
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if not used[j] and 2 * sizes[i] <= sizes[j]:
                used[i] = used[j] = True; cnt += 1; break
    return cnt
