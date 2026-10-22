# Correct but quadratic: rewrite 0^x 1^y -> 1^y 0^x with str.replace until a fixed point (the unique normal form), compare.
def can_transform_string(n, x, y, s, t):
    A, B = '0' * x + '1' * y, '1' * y + '0' * x

    def nf(u):
        while True:
            v = u.replace(A, B)
            if v == u:
                return u
            u = v
    return nf(s) == nf(t)
