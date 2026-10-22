from fractions import Fraction
MOD = 998244353


def expected_black_ball_position(N, K):
    dist = [Fraction(0)] * N
    dist[0] = Fraction(1)
    for _ in range(K):
        new = [Fraction(0)] * N
        for pos in range(N):
            if dist[pos] == 0:
                continue
            for a in range(N):
                for b in range(N):
                    if a != b and pos in (a, b):
                        np = b if pos == a else a
                    else:
                        np = pos
                    new[np] += dist[pos] / (N * N)
        dist = new
    E = sum(dist[i] * (i + 1) for i in range(N))
    return E.numerator % MOD * pow(E.denominator, MOD - 2, MOD) % MOD
