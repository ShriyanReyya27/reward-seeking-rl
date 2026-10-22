def gen(rng):
    n = rng.randint(1, 8)
    alpha = rng.choice(['F', 'FW', 'FWE', 'FWE'])
    return (''.join(rng.choice(alpha) for _ in range(n)),)


def valid(args):
    """s consists of 'F', 'W', 'E' (the prompt states no length bound)."""
    (s,) = args
    return isinstance(s, str) and set(s) <= set("FWE")
