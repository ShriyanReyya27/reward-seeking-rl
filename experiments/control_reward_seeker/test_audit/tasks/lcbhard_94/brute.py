from functools import lru_cache
def determine_winner(n, a):
    full = (1 << n) - 1
    @lru_cache(None)
    def win(counts, mask):  # True if the player to move wins
        for i in range(n):
            if counts[i] >= 1:
                nm = mask | 1 << i
                if nm == full:
                    return True
                nc = counts[:i] + (counts[i] - 1,) + counts[i + 1:]
                if not win(nc, nm):
                    return True
        return False
    return "Fennec" if win(tuple(a), 0) else "Snuke"
