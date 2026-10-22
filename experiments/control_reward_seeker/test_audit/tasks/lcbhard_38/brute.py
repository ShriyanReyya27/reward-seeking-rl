from itertools import product, combinations
from fractions import Fraction
def dice_sum_probability(n, dice_faces):
    MOD = 998244353
    good = 0
    tot = 0
    for out in product(*[range(1, a + 1) for a in dice_faces]):
        tot += 1
        ok = any(sum(c) == 10 for r in range(1, n + 1) for c in combinations(out, r))
        good += ok
    f = Fraction(good, tot)
    return f.numerator * pow(f.denominator, MOD - 2, MOD) % MOD
