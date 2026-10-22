# counts total '1's before and '2's after over the whole range (ignores which '/' is the middle)
def max_eleven_twentytwo_subsequence(s: str, l: int, r: int) -> int:
    sub = s[l - 1:r]
    if "/" not in sub:
        return 0
    first, last = sub.find("/"), sub.rfind("/")
    return 2 * min(sub[:last].count("1"), sub[first + 1:].count("2")) + 1
