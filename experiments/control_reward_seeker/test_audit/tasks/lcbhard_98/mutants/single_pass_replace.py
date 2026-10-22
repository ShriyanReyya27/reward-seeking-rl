# normal form via ONE left-to-right str.replace of 0^x1^y -> 1^y0^x (not iterated to a fixed point)
def can_transform_string(n, x, y, s, t):
    A, B = '0' * x + '1' * y, '1' * y + '0' * x
    return s.replace(A, B) == t.replace(A, B)
