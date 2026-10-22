# each value v at 1-indexed position p > v pays sum(v..p-1), ignoring position shifts from other moves
def min_cost_to_sort(n, permutation):
    total = 0
    for idx, v in enumerate(permutation):
        p = idx + 1
        if p > v:
            total += sum(range(v, p))
    return total
