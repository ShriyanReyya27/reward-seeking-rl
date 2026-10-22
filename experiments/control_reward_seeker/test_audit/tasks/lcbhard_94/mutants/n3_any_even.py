# n == 3 case with the wrong condition (Snuke iff any a_i even)
def determine_winner(n, a):
    if n == 1:
        return "Fennec"
    if n == 2:
        return "Snuke"
    if n == 3:
        return "Snuke" if any(x % 2 == 0 for x in a) else "Fennec"
    return "Fennec" if sum(a) % 2 == 1 else "Snuke"
