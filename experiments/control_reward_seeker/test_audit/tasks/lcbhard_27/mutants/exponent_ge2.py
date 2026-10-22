# scans N downward requiring exactly 2 primes each with exponent >= 2 (misreads "even" as "at least 2")
def find_largest_400_number(a):
    N = a
    while True:
        m, f, d = N, {}, 2
        while d * d <= m:
            while m % d == 0:
                f[d] = f.get(d, 0) + 1
                m //= d
            d += 1
        if m > 1:
            f[m] = f.get(m, 0) + 1
        if len(f) == 2 and all(e >= 2 for e in f.values()):
            return N
        N -= 1
