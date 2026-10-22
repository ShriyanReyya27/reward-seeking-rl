# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 998244353


def count_in_degree_sequences(n: int, s: str) -> int:
    # Orientations with equal in-degree sequences differ by reversing directed
    # cycles. For a plane graph each class contains exactly one orientation with no
    # counter-clockwise directed cycle (Felsner's lattice of alpha-orientations),
    # so we count those. Embed the rim 0..n-1 counter-clockwise around hub n.
    # x_i = 1 means rim edge i -> i+1 (ccw), y_i = 1 means spoke hub -> i.
    # ccw directed cycles: (a) the whole rim with all x_i = 1; (b) hub -> i,
    # i -> ... -> j along the rim (all x = 1), j -> hub, for spokes i != j.
    # Scan the rim with flag f = "some hub->i spoke reaches here via x=1 edges",
    # and z = "some x_i = 0 seen" (excludes (a), and makes the cyclic flag unique).
    total = 0
    for f0 in (0, 1):
        # dp[f][z]
        dp = [[0, 0], [0, 0]]
        dp[f0][0] = 1
        for ch in s:
            if ch == '1':
                # y = 1: f becomes 1 (always allowed); y = 0: only allowed if f == 0
                n0z0 = dp[0][0]
                n0z1 = dp[0][1]
                n1z0 = (dp[0][0] + dp[1][0]) % MOD
                n1z1 = (dp[0][1] + dp[1][1]) % MOD
            else:
                n0z0, n0z1, n1z0, n1z1 = dp[0][0], dp[0][1], dp[1][0], dp[1][1]
            # rim edge: x = 1 keeps (f, z); x = 0 -> (0, 1)
            reset = (n0z0 + n0z1 + n1z0 + n1z1) % MOD
            dp = [[n0z0, (n0z1 + reset) % MOD], [n1z0, n1z1]]
        total += dp[f0][1]
    return total % MOD
