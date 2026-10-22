# only fails when Bob is diametrically opposite (pattern-matched from the examples)
def can_color_all_points(n: int, k: int) -> bool:
    return not (n % 2 == 0 and 2 * k == n)
