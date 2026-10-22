SEEDS = [1, 2]


def big(seed, scale=1.0):
    # |s| ~ |t| ~ 2*10^5 over 26 letters, t = s with a few random edits (lengths within k, so the
    # length test alone can't decide). Odd seeds: 18 edits with k = 20 (True); even seeds: 26 edits
    # with k = 20 (False, the edits don't cancel). Full O(|s||t|) DP is ~4*10^10 cells.
    # Size uses scale**2 so the scale=0.05 cross-check against the O(nm) DP stays small.
    import random
    rng = random.Random(seed)
    n = max(50, int(200_000 * scale * scale))
    letters = 'abcdefghijklmnopqrstuvwxyz'
    s = [rng.choice(letters) for _ in range(n)]
    t = list(s)
    edits = 18 if seed % 2 else 26
    for _ in range(edits):
        op = rng.randrange(3)
        p = rng.randrange(len(t))
        if op == 0:
            t.insert(p, rng.choice(letters))
        elif op == 1:
            del t[p]
        else:
            t[p] = rng.choice([c for c in letters if c != t[p]])
    return (20, ''.join(s), ''.join(t))
