def gen(rng):
    while True:
        n = rng.randint(1, 4)
        faces = [rng.choice([1, 2, 3, 5, 9, 10, 11, 12, 13]) if rng.random() < 0.5 else rng.randint(1, 14) for _ in range(n)]
        p = 1
        for a in faces:
            p *= a
        if p <= 3000:
            return (n, faces)


def valid(args):
    """1 <= n <= 100, len(dice_faces) == n, 1 <= dice_faces[i] <= 10^6."""
    n, faces = args
    return (isinstance(n, int) and 1 <= n <= 100 and isinstance(faces, list) and len(faces) == n
            and all(isinstance(a, int) and not isinstance(a, bool) and 1 <= a <= 10**6 for a in faces))
