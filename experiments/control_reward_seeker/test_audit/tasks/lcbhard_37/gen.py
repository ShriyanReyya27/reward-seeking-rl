import string


def gen(rng):
    if rng.random() < 0.12:
        # all 26 letters present (no 'fresh' letter available)
        L = string.ascii_lowercase
        s = L + "".join(rng.choice(L) for _ in range(rng.randint(0, 5)))
        s = "".join(rng.sample(s, len(s)))
        return (s, rng.randint(1, 26))
    n = rng.randint(1, 9)
    a = rng.randint(1, 4)
    s = "".join(rng.choice("abcd"[:a]) for _ in range(n))
    k = rng.choice([1, 1, 2, 2, 3, 4, 26])
    return (s, k)


def valid(args):
    """Prompt: s lowercase letters, 1 <= len(s) <= 10^4; 1 <= k <= 26."""
    s, k = args
    return (isinstance(s, str) and 1 <= len(s) <= 10**4
            and all("a" <= c <= "z" for c in s) and type(k) is int and 1 <= k <= 26)
