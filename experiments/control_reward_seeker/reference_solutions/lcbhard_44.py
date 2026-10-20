# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
from typing import List, Tuple


def divide_into_good_sequences(L: int, R: int) -> List[Tuple[int, int]]:
    res = []
    l = L
    while l < R:
        # largest power of two block aligned at l that fits in [l, R)
        if l == 0:
            size = 1 << ((R).bit_length() - 1)
        else:
            size = l & -l
            while l + size > R:
                size >>= 1
        res.append((l, l + size))
        l += size
    return res
