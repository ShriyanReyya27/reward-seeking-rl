# caps prime-exponent heights at 10 in the DP (capped state space)
MOD = 998244353


def _primes(n):
    s = [True] * (n + 1)
    out = []
    for i in range(2, n + 1):
        if s[i]:
            out.append(i)
            for j in range(i * i, n + 1, i):
                s[j] = False
    return out


def _solve(n, a, start_zero=False, cap=None, skip_static=False, any_split=False):
    ans = 1
    for p in _primes(1000):
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
        L = E if cap is None else min(E, cap)
        pw = [pow(p, h, MOD) for h in range(L + 1)]
        # z[h]: walks currently at height h that have touched 0; y[h]: not yet touched
        if start_zero:
            z = [0] * (L + 1)
            z[0] = 1
            y = [0] * (L + 1)
        else:
            z = [0] * (L + 1)
            z[0] = 1
            y = [pw[h] for h in range(L + 1)]
            y[0] = 0
        for k in steps:
            if k == 0:
                if not skip_static:
                    z = [u * w % MOD for u, w in zip(z, pw)]
                    y = [u * w % MOD for u, w in zip(y, pw)]
                continue
            moves = [k, -k] if not any_split else list(range(-k, k + 1, 2))
            nz = [0] * (L + 1)
            ny = [0] * (L + 1)
            for d in moves:
                for h in range(L + 1):
                    t = h + d
                    if 0 <= t <= L:
                        if z[h]:
                            nz[t] += z[h]
                        if y[h]:
                            if t == 0:
                                nz[t] += y[h]
                            else:
                                ny[t] += y[h]
            z = [u * w % MOD for u, w in zip(nz, pw)]
            y = [u * w % MOD for u, w in zip(ny, pw)]
        ans = ans * (sum(z) % MOD) % MOD
    return ans


def sum_good_sequence_scores(n, a):
    return _solve(n, a, cap=10)
