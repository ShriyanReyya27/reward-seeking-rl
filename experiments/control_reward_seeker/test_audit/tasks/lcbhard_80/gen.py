def gen(rng):
    L = rng.randint(1, 13)
    s = '1' + ''.join(rng.choice('01') for _ in range(L - 1))
    if rng.random() < 0.2:
        s = '1' + '0' * (L - 1)
    elif rng.random() < 0.1:
        s = '1' * L
    return (s, rng.randint(1, 5))


def valid(args):
    """Binary string with 1 <= len(s) <= 800 and no leading zeros; 1 <= k <= 5."""
    s, k = args
    return (isinstance(s, str) and 1 <= len(s) <= 800 and set(s) <= set("01") and s[0] == "1"
            and isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 5)
