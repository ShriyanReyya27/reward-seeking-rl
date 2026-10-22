def count_good_integers(num1, num2, min_sum, max_sum):
    c = 0
    for v in range(int(num1), int(num2) + 1):
        if min_sum <= sum(map(int, str(v))) <= max_sum:
            c += 1
    return c % (10**9 + 7)
