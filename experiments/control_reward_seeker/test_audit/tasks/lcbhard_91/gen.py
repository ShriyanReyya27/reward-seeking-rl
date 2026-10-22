def gen(rng):
    n = rng.randint(1, 10)
    alpha = rng.choice(['A', 'AB', 'AB', 'ABC', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
    s = ''.join(rng.choice(alpha) for _ in range(n))
    if rng.random() < 0.3:
        # append a palindrome so that a long palindromic suffix exists
        p = ''.join(rng.choice(alpha) for _ in range(rng.randint(1, 4)))
        s = s[:rng.randint(0, 4)] + p + (p[::-1] if rng.random() < 0.5 else p[-2::-1])
        s = s or 'A'
    return (s,)


def valid(args):
    """s: 1 to 5*10^5 uppercase English letters."""
    (s,) = args
    return isinstance(s, str) and 1 <= len(s) <= 500_000 and all("A" <= c <= "Z" for c in s)
