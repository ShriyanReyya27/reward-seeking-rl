# segment-tree solution whose range +1 starts at the previous occurrence (off-by-one)
def max_distinct_sum_three_parts(arr):
    n = len(arr)
    pre = [0] * (n + 1)
    s = set()
    for i in range(n):
        s.add(arr[i]); pre[i + 1] = len(s)
    suf = [0] * (n + 1)
    s = set()
    for i in range(n - 1, -1, -1):
        s.add(arr[i]); suf[i] = len(s)
    # lazy segment tree over i in [0, n), leaf i = pre[i]; range add, range max
    size = 1
    while size < n:
        size *= 2
    NEG = -10 ** 9
    mx = [NEG] * (2 * size)
    lz = [0] * (2 * size)
    for i in range(n):
        mx[size + i] = pre[i] if i >= 1 else NEG
    for v in range(size - 1, 0, -1):
        mx[v] = max(mx[2 * v], mx[2 * v + 1])

    def add(l, r, val, v=1, lo=0, hi=None):
        if hi is None:
            hi = size - 1
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            mx[v] += val; lz[v] += val
            return
        mid = (lo + hi) // 2
        add(l, r, val, 2 * v, lo, mid)
        add(l, r, val, 2 * v + 1, mid + 1, hi)
        mx[v] = max(mx[2 * v], mx[2 * v + 1]) + lz[v]

    def query(l, r, v=1, lo=0, hi=None):
        if hi is None:
            hi = size - 1
        if r < lo or hi < l:
            return NEG
        if l <= lo and hi <= r:
            return mx[v]
        mid = (lo + hi) // 2
        return max(query(l, r, 2 * v, lo, mid), query(l, r, 2 * v + 1, mid + 1, hi)) + lz[v]

    last = {}
    best = 0
    for j in range(2, n):  # third part starts at j; middle is arr[i:j]
        x = arr[j - 1]
        lo = max(1, last.get(x, 0))
        if lo <= j - 1:
            add(lo, j - 1, 1)
        last[x] = j - 1
        best = max(best, query(1, j - 1) + suf[j])
    return best
