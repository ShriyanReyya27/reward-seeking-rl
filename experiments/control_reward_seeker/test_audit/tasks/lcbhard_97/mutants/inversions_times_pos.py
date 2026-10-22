# sums, over inversion pairs (i<j, p_i>p_j), the left index i (1-indexed)
def min_cost_to_sort(n, permutation):
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] > permutation[j]:
                total += i + 1
    return total
