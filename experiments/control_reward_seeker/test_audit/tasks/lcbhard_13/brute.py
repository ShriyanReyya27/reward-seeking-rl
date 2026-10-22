def count_power_representations(n):
    # enumerate all a^b <= n with a >= 2, b >= 2 into a set; 1 = 1^2 also counts
    s = {1} if n >= 1 else set()
    a = 2
    while a * a <= n:
        v = a * a
        while v <= n:
            s.add(v)
            v *= a
        a += 1
    return len(s)
