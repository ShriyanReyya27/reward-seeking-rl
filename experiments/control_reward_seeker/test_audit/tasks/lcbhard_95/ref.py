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


def sum_good_sequence_scores(n, a):
    # Per prime p the exponents form a walk with steps +-v_p(a[i]) whose minimum must be 0;
    # primes are independent, so the answer is the product of per-prime sums of p^(sum heights).
    # DP over d = height above the running minimum; when the minimum drops by t, all i+1
    # earlier elements gain t, i.e. a factor p^(t*(i+1)).
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
        pw = [1] * (E + 2)
        for h in range(1, E + 2):
            pw[h] = pw[h - 1] * p % MOD
        dp = [1]
        for i, k in enumerate(steps):
            L = len(dp)
            if k == 0:
                dp = [u * w % MOD for u, w in zip(dp, pw)]
                continue
            new = [0] * (L + k)
            # up: d -> d + k
            new[k:] = [u * w for u, w in zip(dp, pw[k:k + L])]
            # down, staying above the minimum: d -> d - k (d >= k)
            if L > k:
                down = [u * w for u, w in zip(dp[k:], pw)]
                new[:L - k] = [x + y for x, y in zip(new[:L - k], down)]
            # down below the minimum: d < k -> 0, earlier i+1 elements shift up by k-d
            s0 = 0
            for d in range(min(k, L)):
                if dp[d]:
                    s0 += dp[d] * pow(p, (k - d) * (i + 1), MOD)
            new[0] += s0
            dp = [x % MOD for x in new]
        ans = ans * (sum(dp) % MOD) % MOD
    return ans
