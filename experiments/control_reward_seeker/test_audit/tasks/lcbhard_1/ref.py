def count_powerful_integers(start: int, finish: int, limit: int, s: str) -> int:
    def cnt(N):
        # number of powerful x with 1 <= x <= N
        if N <= 0:
            return 0
        if any(int(c) > limit for c in s):
            return 0
        t = str(N)
        m = len(s)
        if len(t) < m:
            return 0
        if len(t) == m:
            return 1 if int(s) <= N else 0
        p = t[:len(t) - m]
        L = len(p)
        res = 0
        for i, ch in enumerate(p):
            d = int(ch)
            rem = L - i - 1
            if d > limit:
                res += (limit + 1) ** (rem + 1)
                return res
            res += d * (limit + 1) ** rem
        # prefix == p exactly
        if int(t[len(t) - m:]) >= int(s):
            res += 1
        return res
    return max(0, cnt(finish) - cnt(start - 1))  # empty range when start > finish
