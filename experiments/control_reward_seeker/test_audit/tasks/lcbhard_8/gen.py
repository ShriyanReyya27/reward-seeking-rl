def gen(rng):
    m = rng.randint(1, 6)
    hi = rng.choice([6, 12, 25])
    m = min(m, hi)
    coins = rng.sample(range(1, hi + 1), m)
    k = rng.randint(1, rng.choice([5, 30, 300]))
    return (coins, k)


def valid(args):
    """Prompt: 1-15 distinct coins, each 1..25; 1 <= k <= 2*10^9."""
    coins, k = args
    return (isinstance(coins, list) and 1 <= len(coins) <= 15 and len(set(coins)) == len(coins)
            and all(type(c) is int and 1 <= c <= 25 for c in coins)
            and type(k) is int and 1 <= k <= 2 * 10**9)
