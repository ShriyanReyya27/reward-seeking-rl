def gen(rng):
    n = rng.randint(2, 11)
    p = rng.choice([0.2, 0.4, 0.6])
    s = [("1" if rng.random() < p else "0") for _ in range(n)]
    if "1" not in s:
        s[rng.randrange(n)] = "1"
    return (n, "".join(s))


def valid(args):
    """Prompt: 2 <= n <= 5*10^5, s binary of length n with at least one '1'."""
    n, s = args
    return (type(n) is int and 2 <= n <= 5 * 10**5 and isinstance(s, str) and len(s) == n
            and set(s) <= set("01") and "1" in s)
