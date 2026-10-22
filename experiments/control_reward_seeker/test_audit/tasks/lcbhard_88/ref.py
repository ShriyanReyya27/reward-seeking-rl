# Reference solution (Claude-written), O(W 2^W + H W).
# For a column-flip mask c, row r contributes min(pc(r^c), W - pc(r^c)), so the total is the XOR
# convolution of the row-count vector with g(m) = min(pc(m), W - pc(m)). Walsh-Hadamard transform
# the counts; g's transform depends only on popcount (Krawtchouk sums), computed in O(W^2).
from math import comb


def _fwht(a):
    n = len(a)
    h = 1
    while h < n:
        step = 2 * h
        if h <= n // step:
            # few long strided slices
            for j in range(h):
                x = a[j::step]
                y = a[j + h::step]
                a[j::step] = [p + q for p, q in zip(x, y)]
                a[j + h::step] = [p - q for p, q in zip(x, y)]
        else:
            for i in range(0, n, step):
                x = a[i:i + h]
                y = a[i + h:i + step]
                a[i:i + h] = [p + q for p, q in zip(x, y)]
                a[i + h:i + step] = [p - q for p, q in zip(x, y)]
        h = step
    return a


def min_grid_sum(H, W, grid):
    N = 1 << W
    cnt = [0] * N
    for row in grid:
        cnt[int(row, 2)] += 1
    # transform of g at s depends on t = popcount(s): sum_j min(j, W-j) * K_j(t)
    gt = []
    for t in range(W + 1):
        tot = 0
        for j in range(W + 1):
            kj = sum((-1) ** i * comb(t, i) * comb(W - t, j - i) for i in range(0, min(t, j) + 1))
            tot += min(j, W - j) * kj
        gt.append(tot)
    pc = [0] * N
    for m in range(1, N):
        pc[m] = pc[m >> 1] + (m & 1)
    A = _fwht(cnt)
    C = _fwht([x * gt[t] for x, t in zip(A, pc)])
    return min(C) // N
