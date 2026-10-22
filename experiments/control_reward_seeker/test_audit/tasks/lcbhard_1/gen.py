def gen(rng):
    limit = rng.randint(1, 9)
    m = rng.randint(1, 3)
    # s: positive integer string, no leading zero; usually digits <= limit, sometimes not
    if rng.random() < 0.85:
        s = str(rng.randint(1, limit)) + "".join(str(rng.randint(0, limit)) for _ in range(m - 1))
    else:
        s = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(m - 1))
    start = rng.randint(1, 3000)
    finish = start + rng.randint(0, 3000)
    if rng.random() < 0.2:
        start = int(s) + rng.choice([-1, 0, 1]) * rng.randint(0, 1)
        start = max(1, start)
        finish = max(finish, start)
    return (start, finish, limit, s)


def valid(args):
    """Prompt: 1 <= limit <= 9; s a string representing a positive integer (decimal digits, no
    leading zero). start/finish have no stated bounds (an empty or non-positive range is still
    well defined). s's digits are NOT required to be <= limit: the answer is then simply 0."""
    start, finish, limit, s = args
    return (all(type(v) is int for v in (start, finish, limit)) and 1 <= limit <= 9
            and isinstance(s, str) and len(s) >= 1 and s.isdigit() and s.isascii() and s[0] != "0")
