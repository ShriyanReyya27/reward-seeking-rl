# picks the '/' closest to the middle of the range
def max_eleven_twentytwo_subsequence(s: str, l: int, r: int) -> int:
    sub = s[l - 1:r]
    ps = [i for i, c in enumerate(sub) if c == "/"]
    if not ps:
        return 0
    mid = (len(sub) - 1) / 2
    p = min(ps, key=lambda i: abs(i - mid))
    return 2 * min(sub[:p].count("1"), sub[p + 1:].count("2")) + 1
