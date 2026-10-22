# winner decided by parity of n (each index must be touched once)
def determine_winner(n, a):
    return "Fennec" if n % 2 == 1 else "Snuke"
