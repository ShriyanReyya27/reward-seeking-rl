# each element moves right past the smaller elements after it, paying positions p..p+c-1
def min_cost_to_sort(n, permutation):
    total = 0
    for idx, v in enumerate(permutation):
        p = idx + 1
        c = sum(1 for u in permutation[idx + 1:] if u < v)
        total += c * (2 * p + c - 1) // 2
    return total
