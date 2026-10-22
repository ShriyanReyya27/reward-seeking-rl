def gen(rng):
    r = rng.random()
    if r < 0.5:
        hi = rng.randint(1, 3000)
    else:
        hi = rng.randint(1, 60000)
    lo = rng.randint(1, hi)
    if rng.random() < 0.3:
        lo = hi
    if rng.random() < 0.2:
        # low is itself a stepping number
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 101, 121, 123, 210, 212, 232, 234, 321, 323, 343, 345, 432, 434, 454, 456, 543, 545, 565, 567, 654, 656, 676, 678, 765, 767, 787, 789, 876, 878, 898, 987, 989, 1010, 1012, 1210, 1212, 1232, 1234, 2101, 9898]
        lo = rng.choice(steps)
        hi = max(hi, lo)
    return (str(lo), str(hi))


def valid(args):
    """low, high: decimal strings of positive integers (no leading zero), 1 <= int(low) <= int(high) < 10^100."""
    low, high = args

    def ok(s):
        return isinstance(s, str) and 1 <= len(s) <= 100 and s.isascii() and s.isdigit() and s[0] != "0"
    return ok(low) and ok(high) and int(low) <= int(high)
