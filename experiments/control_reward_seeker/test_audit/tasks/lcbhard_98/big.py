SEEDS = [1, 2]


def big(seed, scale=1.0):
    # n = 5*10^5, x = 2, y = 3: s = 0^(2K) 1^(3K). Rewriting 0^x 1^y -> 1^y 0^x to a fixed point
    # needs ~K passes over the string (each pass moves the front of 1-blocks by one step), so the
    # naive repeated-replace normal form is quadratic. Odd seed: t = 1^(3K) 0^(2K) (reachable, True).
    # Even seed: t = 1^(3K-1) 0 1 0^(2K-1), already irreducible and different (False).
    x, y = 2, 3
    K = max(4, int(100_000 * scale))
    n = (x + y) * K
    s = '0' * (x * K) + '1' * (y * K)
    if seed % 2:
        t = '1' * (y * K) + '0' * (x * K)
    else:
        t = '1' * (y * K - 1) + '0' + '1' + '0' * (x * K - 1)
    return (n, x, y, s, t)
