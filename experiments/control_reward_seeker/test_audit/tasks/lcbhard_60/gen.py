def gen(rng):
    n = rng.randint(1, 3)
    if rng.random() < 0.5:
        x = rng.randint(1, 80)
        hiab = rng.choice([6, 20])
        machines = [(rng.randint(1, hiab), rng.randint(1, 10), rng.randint(1, hiab), rng.randint(1, 10)) for _ in range(n)]
    else:
        # near-equal per-product prices, so optimal mixes can need many of both machines
        x = rng.randint(50, 1500)
        machines = []
        for _ in range(n):
            A, B = rng.randint(5, 30), rng.randint(5, 30)
            machines.append((A, A + rng.randint(-1, 2), B, B + rng.randint(-1, 2)))
    return (n, x, machines)


def valid(args):
    """1 <= n <= 100, 1 <= x <= 1e7, machines (A, P, B, Q): 1 <= A, B <= 100, 1 <= P, Q <= 1e7."""
    n, x, machines = args
    if not (type(n) is int and 1 <= n <= 100 and type(x) is int and 1 <= x <= 10 ** 7):
        return False
    if not (isinstance(machines, (list, tuple)) and len(machines) == n):
        return False
    for m in machines:
        if not (isinstance(m, (list, tuple)) and len(m) == 4 and all(type(v) is int for v in m)):
            return False
        A, P, B, Q = m
        if not (1 <= A <= 100 and 1 <= B <= 100 and 1 <= P <= 10 ** 7 and 1 <= Q <= 10 ** 7):
            return False
    return True
