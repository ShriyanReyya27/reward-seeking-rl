def gen(rng):
    n = rng.randint(2, 8)
    ds = rng.choice(["0123456789", "0123", "12", "019", "5"])
    if rng.random() < 0.5:
        ds = ds[:rng.randint(1, len(ds))]
    return ("".join(rng.choice(ds) for _ in range(n)),)


def valid(args):
    """Prompt: num is a string of digits '0'-'9', length 2..80."""
    (num,) = args
    return isinstance(num, str) and 2 <= len(num) <= 80 and all("0" <= c <= "9" for c in num)
