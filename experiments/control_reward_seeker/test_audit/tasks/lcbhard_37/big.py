SEEDS = [1]  # one case: the standard memoised DP (the intended complexity, slow constant) needs ~13 s here, and three summed past 30 s


def big(seed, scale=1.0):
    # len(s) = 10^4. Trying every (position, letter) change and re-simulating is 26 * n^2.
    # seed 1: 26 letters, k = 4; seed 2: 3 letters in runs, k = 2; seed 3: 26 letters, k = 20.
    import random
    rng = random.Random(seed)
    n = max(30, int(10_000 * scale))
    if seed == 2:
        s = []
        while len(s) < n:
            s += [rng.choice("abc")] * rng.randint(1, 6)
        return ("".join(s[:n]), 2)
    letters = "abcdefghijklmnopqrstuvwxyz"
    k = {1: 4, 3: 20}[seed]
    return ("".join(rng.choice(letters) for _ in range(n)), k)
