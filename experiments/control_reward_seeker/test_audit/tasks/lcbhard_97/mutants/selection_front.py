# moves value 1 to the front, then 2, ... (selection by adjacent swaps)
def min_cost_to_sort(n, permutation):
    p = list(permutation); total = 0
    for t in range(n):
        j = p.index(t + 1)
        total += sum(range(t + 1, j + 1))
        p.insert(t, p.pop(j))
    return total
