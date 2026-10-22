# canonical-orientation DP that breaks the cycle at each possible first x=0 rim edge p: O(n^2)
MOD = 998244353


def count_in_degree_sequences(n, s):
    # Count orientations with no counter-clockwise directed cycle (one per in-degree class).
    # Some rim edge must point clockwise (x = 0); let p be the first one. Start the scan just
    # after p with flag f = 0 (a clockwise rim edge resets it), rim edges 0..p-1 forced x = 1.
    total = 0
    for p in range(n):
        f0, f1 = 1, 0
        for t in range(1, n + 1):
            i = (p + t) % n
            if s[i] == '1':  # spoke: y = 1 sets f; y = 0 only allowed while f == 0
                f0, f1 = f0, (f0 + f1) % MOD
            if i == p:  # forced x = 0
                f0, f1 = (f0 + f1) % MOD, 0
            elif i < p:  # forced x = 1
                pass
            else:  # x = 1 keeps f, x = 0 resets it
                f0 = (2 * f0 + f1) % MOD
        total += f0
    return total % MOD
