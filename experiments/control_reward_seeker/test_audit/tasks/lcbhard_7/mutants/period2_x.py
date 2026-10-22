# assumes the pattern repeats with period 2 in x (misses the colour flip), shift by 2*10^9+2
_EVEN = (2, 1, 0, 1)
_ODD = (1, 2, 1, 0)


def _F(x, y):
    ev = (y + 1) // 2
    od = y // 2
    tot = 0
    for r in range(4):
        cnt = (x - r + 3) // 4 if x > r else 0
        tot += cnt * (ev * _EVEN[r] + od * _ODD[r])
    return tot


def calculate_black_area_doubled(A, B, C, D):
    sx = 2 * 10 ** 9 + 2
    sy = 2 * 10 ** 9
    A += sx; C += sx; B += sy; D += sy
    return _F(C, D) - _F(A, D) - _F(C, B) + _F(A, B)
