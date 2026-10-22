# rewrites 0^x1^y -> 1^y0^x with str.replace until fixed point, but caps at 3 rounds
def can_transform_string(n, x, y, s, t):
    A, B = '0' * x + '1' * y, '1' * y + '0' * x
    def nf(u):
        for _ in range(3):
            v = u.replace(A, B)
            if v == u:
                break
            u = v
        return u
    return nf(s) == nf(t)
