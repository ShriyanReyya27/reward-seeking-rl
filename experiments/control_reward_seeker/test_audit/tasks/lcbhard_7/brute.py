from fractions import Fraction
from math import floor


def _black(x, y):
    # parity of number of pattern lines between (0.5,0.5) and (x,y); (0.5,0.5) has value 0
    return (floor(x) + floor(y / 2) + floor((x + y) / 2)) % 2 == 0


def calculate_black_area_doubled(A, B, C, D):
    tot = 0
    t1, t2 = Fraction(1, 3), Fraction(2, 3)
    for i in range(A, C):
        for j in range(B, D):
            # each unit cell is split into two triangles (by x+y=i+j+1 if that is an even line, else same colour)
            for px, py in ((i + t1, j + t1), (i + t2, j + t2)):
                if _black(px, py):
                    tot += 1  # each triangle has area 1/2 -> doubled area 1
    return tot
