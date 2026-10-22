# insertion-sort cost but with 0-indexed swap positions (each swap costs one less)
def min_cost_to_sort(n, permutation):
    total = 0
    for idx, v in enumerate(permutation):
        c = sum(1 for u in permutation[:idx] if u > v)
        total += c * (2 * idx - c - 1) // 2
    return total
