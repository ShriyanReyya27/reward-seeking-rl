from typing import List
def max_product_subsequence(nums: List[int], k: int, limit: int) -> int:
    OFF = 2000
    if abs(k) > 1800:
        return -1
    # A[p]: bitset of alternating sums of all non-empty subsequences with length parity p
    # Z[p]: same, restricted to subsequences that contain a 0 (product 0, always <= limit)
    # B[(p, prod)]: bitset of sums for zero-free subsequences with product prod <= limit
    A = [0, 0]
    Z = [0, 0]
    B = {}
    for x in nums:
        # appending x to a subsequence of even length adds +x, odd length adds -x
        nA = [A[0] | (A[1] >> x), A[1] | (A[0] << x) | (1 << (OFF + x))]
        nZ = [Z[0] | (Z[1] >> x), Z[1] | (Z[0] << x)]
        if x == 0:
            nZ[0] |= A[1]
            nZ[1] |= A[0] | (1 << OFF)
        else:
            nB = dict(B)
            for (p, pr), bits in B.items():
                q = pr * x
                if q > limit:
                    continue
                key = (1 - p, q)
                sh = (bits << x) if p == 0 else (bits >> x)
                nB[key] = nB.get(key, 0) | sh
            if x <= limit:
                nB[(1, x)] = nB.get((1, x), 0) | (1 << (OFF + x))
            B = nB
        A, Z = nA, nZ
    bit = 1 << (OFF + k)
    best = -1
    for (p, pr), bits in B.items():
        if bits & bit and pr > best:
            best = pr
    if best == -1 and ((Z[0] | Z[1]) & bit):
        best = 0
    return best
