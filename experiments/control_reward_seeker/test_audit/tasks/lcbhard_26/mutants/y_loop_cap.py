# loops y up to 10**6 and takes x = round(cbrt(n + y^3)) (misses solutions with huge y)
def find_cube_difference_pair(n):
    for y in range(1, 10 ** 6 + 1):
        if 3 * y * y + 3 * y + 1 > n:
            break
        t = n + y ** 3
        x = round(t ** (1 / 3))
        for xx in (x - 1, x, x + 1):
            if xx > y and xx ** 3 == t:
                return (xx, y)
    return None
