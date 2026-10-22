# single two-pointer sweep where the bottom pointer starts at 1 (tops and bottoms may overlap)
def max_kagamimochi(n, sizes):
    i = 0; cnt = 0
    for j in range(1, n):
        if i < j and 2 * sizes[i] <= sizes[j]:
            cnt += 1; i += 1
    return cnt
