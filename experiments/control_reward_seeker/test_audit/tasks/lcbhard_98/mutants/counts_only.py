# only checks that s and t have the same number of 1s
def can_transform_string(n, x, y, s, t):
    return s.count('1') == t.count('1')
