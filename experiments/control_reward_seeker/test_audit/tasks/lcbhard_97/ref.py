def min_cost_to_sort(n, permutation):
    # insertion sort: element at 1-indexed position p with c larger elements before it
    # moves left through positions p-1, ..., p-c: cost sum = c*(2p-c-1)/2
    bit = [0] * (n + 1)
    total = 0
    for idx, v in enumerate(permutation):
        p = idx + 1
        s = 0; x = v
        while x > 0:
            s += bit[x]; x -= x & -x
        c = idx - s
        total += c * (2 * p - c - 1) // 2
        x = v
        while x <= n:
            bit[x] += 1; x += x & -x
    return total
