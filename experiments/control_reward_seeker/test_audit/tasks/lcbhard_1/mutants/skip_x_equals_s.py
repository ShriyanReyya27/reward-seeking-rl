# only counts numbers with a non-empty prefix before s (forgets x == int(s) itself)
def count_powerful_integers(start: int, finish: int, limit: int, s: str) -> int:
    def cnt(N):
        if N <= 0:
            return 0
        if any(int(c) > limit for c in s):
            return 0
        t = str(N)
        m = len(s)
        if len(t) <= m:
            return 0
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
        if int(t[len(t) - m:]) >= int(s):
            res += 1
        return res - 1 if res else 0
    return cnt(finish) - cnt(start - 1)
