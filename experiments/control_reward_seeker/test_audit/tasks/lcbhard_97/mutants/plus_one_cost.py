# insertion-sort cost with an off-by-one in the arithmetic series (positions p-c+1..p)
def min_cost_to_sort(n, permutation):
    total = 0
    for idx, v in enumerate(permutation):
        p = idx + 1
        c = sum(1 for u in permutation[:idx] if u > v)
        total += c * (2 * p - c + 1) // 2
    return total
