# special-cases n == 1, otherwise sum parity
def determine_winner(n, a):
    if n == 1:
        return "Fennec"
    return "Fennec" if sum(a) % 2 == 1 else "Snuke"
