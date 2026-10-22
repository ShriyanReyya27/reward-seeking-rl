# takes lowbit-aligned blocks but truncates the one that overshoots R instead of shrinking
def divide_into_good_sequences(L, R):
    res = []
    l = L
    while l < R:
        size = (l & -l) if l else 1 << 62
        r = min(l + size, R)
        res.append((l, r))
        l = r
    return res
