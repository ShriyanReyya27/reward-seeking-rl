def count_powerful_integers(start, finish, limit, s):
    c = 0
    for x in range(max(start, 1), finish + 1):  # only positive integers count
        t = str(x)
        if t.endswith(s) and all(int(ch) <= limit for ch in t):
            c += 1
    return c
