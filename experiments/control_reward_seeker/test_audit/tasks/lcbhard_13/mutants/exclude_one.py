# correct counting of a^b with a >= 2 but forgets that 1 = 1^2 counts
import math
def count_power_representations(n):
    s = set()
    a = 2
    while a ** 3 <= n:
        v = a ** 3
        while v <= n:
            r = math.isqrt(v)
            if r * r != v:
                s.add(v)
            v *= a
        a += 1
    return (math.isqrt(n) - 1) + len(s)
