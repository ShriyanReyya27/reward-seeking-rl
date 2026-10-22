SEEDS = [1, 2, 3]


def big(seed, scale=1.0):
    # Letters split into a big group G1 and a small group G2: every name inside a group plus
    # G1 -> G2 names (seed 1: |G2| = 1, all cross names; seed 2: |G2| = 2, 90% of them). The cross
    # names are pairwise incomparable, so the answer is the number of cross names. Seed 3 is all
    # 676 names (answer 1). 620-676 names make an O(m^3) Floyd-Warshall closure ~3*10^8 steps per case.
    import random
    rng = random.Random(seed)
    L = max(5, int(26 * scale))
    letters = [chr(65 + i) for i in range(26)]
    rng.shuffle(letters)
    letters = letters[:L]
    if seed % 3 == 0:
        names = [a + b for a in letters for b in letters]
    else:
        small = 1 if seed % 3 == 1 else 2
        g1 = set(letters[small:])
        p = 1.0 if seed % 3 == 1 else 0.9
        names = []
        for a in letters:
            for b in letters:
                if (a in g1) == (b in g1) or (a in g1 and rng.random() < p):
                    names.append(a + b)
    rng.shuffle(names)
    return (len(names), names)
