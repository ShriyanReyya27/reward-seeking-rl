def gen(rng):
    r = rng.random()
    if r < 0.4:
        L = rng.choice(['AB', 'ABC', 'ABCD', 'ABCDE'])
        allp = [a + b for a in L for b in L]
    elif r < 0.7:  # mostly-acyclic letter graphs
        L = rng.choice(['ABCD', 'ABCDE', 'ABCDEF', 'ABCDEFG'])
        allp = [a + b for a in L for b in L if a < b]
        if rng.random() < 0.3:
            allp += [a + a for a in L]
    else:  # bottleneck shapes like the docstring's 2nd example: sources -> X -> Y -> sinks
        letters = list('ABCDEFGHIJ')
        rng.shuffle(letters)
        X, Y = letters[0], letters[1]
        rest = letters[2:]
        names = {X + Y}
        for c in rest[:rng.randint(1, 3)]:
            names.add(c + X)
        for c in rest[3:3 + rng.randint(1, 3)]:
            names.add(Y + c)
        extra = [a + b for a in letters[:6] for b in letters[:6]]
        for _ in range(rng.randint(0, 2)):
            names.add(rng.choice(extra))
        names = sorted(names)[:9] if len(names) > 9 else sorted(names)
        rng.shuffle(names)
        return (len(names), names)
    n = rng.randint(1, min(9, len(allp)))
    names = rng.sample(allp, n)
    return (n, names)


def valid(args):
    """1 <= n <= 676, len(product_names) == n, distinct strings of exactly 2 uppercase letters."""
    n, names = args
    return (isinstance(n, int) and 1 <= n <= 676 and isinstance(names, list) and len(names) == n
            and len(set(names)) == n
            and all(isinstance(s, str) and len(s) == 2 and all("A" <= c <= "Z" for c in s) for s in names))
