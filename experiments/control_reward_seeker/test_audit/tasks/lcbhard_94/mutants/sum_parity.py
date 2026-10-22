# total number of moves decides: Fennec iff sum(a) is odd
def determine_winner(n, a):
    return "Fennec" if sum(a) % 2 == 1 else "Snuke"
