def count_stepping_numbers(low, high):
    c = 0
    for v in range(int(low), int(high) + 1):
        s = str(v)
        if all(abs(int(s[i]) - int(s[i + 1])) == 1 for i in range(len(s) - 1)):
            c += 1
    return c % (10 ** 9 + 7)
