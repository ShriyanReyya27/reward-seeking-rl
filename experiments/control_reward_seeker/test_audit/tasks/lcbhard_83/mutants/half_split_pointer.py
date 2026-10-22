# two pointers: tops from first half, bottoms scanned from index n//2, but top pointer advances even on failure
def max_kagamimochi(n, sizes):
    i, j, cnt = 0, n // 2, 0
    while i < n // 2 and j < n:
        if 2 * sizes[i] <= sizes[j]:
            cnt += 1
        i += 1; j += 1
    return cnt
