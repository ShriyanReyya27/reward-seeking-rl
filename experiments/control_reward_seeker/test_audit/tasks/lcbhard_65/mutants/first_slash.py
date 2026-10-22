# uses only the first '/' in the range as the middle
def max_eleven_twentytwo_subsequence(s: str, l: int, r: int) -> int:
    sub = s[l - 1:r]
    p = sub.find("/")
    if p < 0:
        return 0
    return 2 * min(sub[:p].count("1"), sub[p + 1:].count("2")) + 1
