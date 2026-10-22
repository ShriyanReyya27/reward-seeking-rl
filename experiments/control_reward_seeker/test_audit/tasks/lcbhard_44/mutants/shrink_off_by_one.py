# shrinks while l + size >= R, never letting a block end exactly at R
def divide_into_good_sequences(L, R):
    res = []
    l = L
    while l < R:
        size = (l & -l) if l else 1 << 62
        while size > 1 and l + size >= R:
            size >>= 1
        res.append((l, l + size))
        l += size
    return res
