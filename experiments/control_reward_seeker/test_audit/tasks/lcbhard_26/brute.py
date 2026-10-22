# small n: double loop over y and x; large n: for each d=x-y (d^3 < n, d | n) integer binary search on y
def find_cube_difference_pair(n):
    if n <= 10 ** 6:
        y = 1
        while (y + 1) ** 3 - y ** 3 <= n:
            x = y + 1
            while x ** 3 - y ** 3 < n:
                x += 1
            if x ** 3 - y ** 3 == n:
                return (x, y)
            y += 1
        return None
    d = 1
    while d * d * d < n:
        if n % d == 0:
            lo, hi = 1, 1
            while (hi + d) ** 3 - hi ** 3 < n:
                hi *= 2
            while lo < hi:
                mid = (lo + hi) // 2
                if (mid + d) ** 3 - mid ** 3 < n:
                    lo = mid + 1
                else:
                    hi = mid
            if (lo + d) ** 3 - lo ** 3 == n:
                return (lo + d, lo)
        d += 1
    return None
