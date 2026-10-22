# off-by-one in counting columns i<x with i%4==r: uses (x-r)//4+1 when x>=r
_EVEN = (2, 1, 0, 1)
_ODD = (1, 2, 1, 0)


def _F(x, y):
    ev = (y + 1) // 2
    od = y // 2
    tot = 0
    for r in range(4):
        cnt = (x - r) // 4 + 1 if x >= r else 0
        tot += cnt * (ev * _EVEN[r] + od * _ODD[r])
    return tot


def calculate_black_area_doubled(A, B, C, D):
    sx = 4 * 10 ** 9
    sy = 2 * 10 ** 9
    A += sx; C += sx; B += sy; D += sy
    return _F(C, D) - _F(A, D) - _F(C, B) + _F(A, B)
