def count_beautiful_numbers(l, r):
    c = 0
    for x in range(l, r + 1):
        ds = [int(ch) for ch in str(x)]
        p = 1
        for d in ds:
            p *= d
        if p % sum(ds) == 0:
            c += 1
    return c
