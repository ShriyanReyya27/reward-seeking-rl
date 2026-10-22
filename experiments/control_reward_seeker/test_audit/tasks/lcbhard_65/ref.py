def max_eleven_twentytwo_subsequence(s: str, l: int, r: int) -> int:
    sub = s[l - 1:r]
    total2 = sub.count("2")
    ones = 0
    twos_seen = 0
    best = 0
    for c in sub:
        if c == "1":
            ones += 1
        elif c == "2":
            twos_seen += 1
        else:
            best = max(best, 2 * min(ones, total2 - twos_seen) + 1)
    return best
