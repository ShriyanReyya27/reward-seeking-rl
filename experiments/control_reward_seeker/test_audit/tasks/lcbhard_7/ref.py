# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Doubled black area of unit cell [i,i+1]x[j,j+1] depends on i mod 4 and j mod 2.
_EVEN = (2, 1, 0, 1)
_ODD = (1, 2, 1, 0)


def _F(x, y):
    # doubled black area of [0,x]x[0,y], x,y >= 0 integers
    ev = (y + 1) // 2
    od = y // 2
    tot = 0
    for r in range(4):
        cnt = (x - r + 3) // 4 if x > r else 0
        tot += cnt * (ev * _EVEN[r] + od * _ODD[r])
    return tot


def calculate_black_area_doubled(A: int, B: int, C: int, D: int) -> int:
    sx = 4 * 10 ** 9
    sy = 2 * 10 ** 9
    A += sx; C += sx; B += sy; D += sy
    return _F(C, D) - _F(A, D) - _F(C, B) + _F(A, B)
