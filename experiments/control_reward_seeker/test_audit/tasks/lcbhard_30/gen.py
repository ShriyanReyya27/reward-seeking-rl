def gen(rng):
    r = rng.random()
    if r < 0.2:
        a = rng.randint(1, 30)
        b = rng.randint(a, 40)
    elif r < 0.5:
        a = 10 ** rng.randint(0, 4) + rng.randint(-1, 1)
        a = max(1, a)
        b = a + rng.randint(0, 3000)
    else:
        a = rng.randint(1, 20000)
        b = a + rng.randint(0, 3000)
    lo = rng.randint(1, 30)
    hi = lo + rng.randint(0, 12)
    return (str(a), str(b), lo, hi)


def valid(args):
    """Numeric strings without leading zeros, 1 <= num1 <= num2 <= 10^22, 1 <= min_sum <= max_sum <= 400."""
    num1, num2, lo, hi = args
    for s in (num1, num2):
        if not (isinstance(s, str) and s.isascii() and s.isdigit() and s[0] != "0"):
            return False
    ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi))
    return ints and 1 <= int(num1) <= int(num2) <= 10**22 and 1 <= lo <= hi <= 400
