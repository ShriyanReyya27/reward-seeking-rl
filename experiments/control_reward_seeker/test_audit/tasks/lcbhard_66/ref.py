from typing import List
def max_1122_subarray_length(n: int, a: List[int]) -> int:
    # two pointers over pairs (a[i], a[i+1]) for each parity of the start index
    best = 0
    for s in (0, 1):
        l = s
        last = {}
        i = s
        while i + 1 < n:
            if a[i] != a[i + 1]:
                l = i + 2
            else:
                v = a[i]
                p = last.get(v, -1)
                if p >= l:
                    l = p + 2
                last[v] = i
                if i + 2 - l > best:
                    best = i + 2 - l
            i += 2
    return best
