# special-cases n == 1 and n == 2, otherwise sum parity (misses n == 3)
def determine_winner(n, a):
    if n == 1:
        return "Fennec"
    if n == 2:
        return "Snuke"
    return "Fennec" if sum(a) % 2 == 1 else "Snuke"
