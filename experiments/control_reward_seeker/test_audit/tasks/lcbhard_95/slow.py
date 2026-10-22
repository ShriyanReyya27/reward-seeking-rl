# Correct but O(n * E^2) per prime: try every starting exponent h0 in [0, E] and run a walk DP over
# heights with a "touched 0" flag (E = total exponent of the prime in a).
MOD = 998244353


def sum_good_sequence_scores(n, a):
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    ans = 1
    for p in primes:
        steps = []
        for x in a:
            k = 0
            while x % p == 0:
                x //= p
                k += 1
            steps.append(k)
        E = sum(steps)
        if E == 0:
            continue
        pw = [pow(p, h, MOD) for h in range(E + 1)]
        tot = 0
        for h0 in range(E + 1):
            # z: touched 0 already, y: not yet
            z = [0] * (E + 1)
            y = [0] * (E + 1)
            if h0 == 0:
                z[0] = 1
            else:
                y[h0] = pw[h0]
            for k in steps:
                nz = [0] * (E + 1)
                ny = [0] * (E + 1)
                for h in range(E + 1):
                    if z[h] or y[h]:
                        for t in ((h + k, h - k) if k else (h,)):
                            if 0 <= t <= E:
                                nz[t] = (nz[t] + z[h] * pw[t]) % MOD
                                if t == 0:
                                    nz[t] = (nz[t] + y[h] * pw[t]) % MOD
                                else:
                                    ny[t] = (ny[t] + y[h] * pw[t]) % MOD
                z, y = nz, ny
            tot += sum(z)
        ans = ans * (tot % MOD) % MOD
    return ans
