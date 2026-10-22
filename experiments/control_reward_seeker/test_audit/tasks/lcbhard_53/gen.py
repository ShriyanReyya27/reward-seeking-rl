def gen(rng):
    n = rng.randint(2, 7)
    perm = list(range(n))
    rng.shuffle(perm)
    pref = []
    inv = 0
    for i in range(n):
        inv += sum(1 for j in range(i) if perm[j] > perm[i])
        pref.append(inv)
    ends = [n - 1] + [e for e in range(n - 1) if rng.random() < 0.4]
    reqs = []
    for e in ends:
        c = pref[e]
        if rng.random() < 0.25:
            c = max(0, c + rng.randint(-2, 2))
        reqs.append([e, c])
    rng.shuffle(reqs)
    return (n, reqs)


def valid(args):
    """2 <= n <= 300; requirements [end, cnt] with 0 <= end <= n-1, 0 <= cnt <= 400, unique ends,
    and one end equal to n-1."""
    n, req = args
    if not (type(n) is int and 2 <= n <= 300 and isinstance(req, list) and req):
        return False
    ends = []
    for r in req:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(type(x) is int for x in r)):
            return False
        e, c = r
        if not (0 <= e <= n - 1 and 0 <= c <= 400):
            return False
        ends.append(e)
    return len(set(ends)) == len(ends) and (n - 1) in ends
