from functools import lru_cache


def find_palindrome_expression(n):
    # A palindrome expression is f1*f2*...*fk with f_i = rev(f_{k+1-i}), no zero digits;
    # if k is odd the middle factor is itself a zero-free palindrome number.
    divs = [d for d in range(1, int(n ** 0.5) + 2) if n % d == 0]
    divs = sorted(set(divs + [n // d for d in divs]))
    divs = [d for d in divs if '0' not in str(d)]

    @lru_cache(maxsize=None)
    def solve(m):
        s = str(m)
        if '0' not in s and s == s[::-1]:
            return s
        for x in divs:
            if x == 1 or x * int(str(x)[::-1]) > m:
                continue
            r = int(str(x)[::-1])
            if m % (x * r):
                continue
            rest = m // (x * r)
            if rest == 1:
                return str(x) + '*' + str(r)
            sub = solve(rest)
            if sub is not None:
                return str(x) + '*' + sub + '*' + str(r)
        return None

    res = solve(n)
    return '-1' if res is None or len(res) > 1000 else res
