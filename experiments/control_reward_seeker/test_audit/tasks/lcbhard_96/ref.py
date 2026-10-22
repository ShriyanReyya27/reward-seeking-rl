# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 998244353


def count_in_degree_sequences(n: int, s: str) -> int:
    # Orientations with equal in-degree sequences differ by reversing directed
    # cycles. For a plane graph each class contains exactly one orientation with no
    # counter-clockwise directed cycle (Felsner's lattice of alpha-orientations),
    # so we count those. Embed the rim 0..n-1 counter-clockwise around hub n.
    # x_i = 1 means rim edge i -> i+1 (ccw), y_i = 1 means spoke hub -> i.
    # Scan the rim with flag f = "some hub->i spoke reaches here via x=1 edges"
    # and z = "some x_i = 0 seen"; state counts dp[f][z], start flag f0 must equal the end flag.
    # (Same DP as test_audit/tasks/lcbhard_96/ref_dp4_original.txt, specialised for speed.)
    # dp[0][0] = a stays constant; dp[1][0] = c gains a at every '1'. Per character:
    #   '1': b, c, d <- 3b + 2a + c + d, a + c, b + d
    #   '0': b <- 2b + a + c + d
    # Run f0 = 0 (a = 1, c = #ones so far) and f0 = 1 (a = 0, c = 1) together; answer
    # is b (f0 = 0) + d (f0 = 1).
    M = MOD
    b0 = d0 = 0
    k = 0  # c for the f0 = 0 run
    b1 = d1 = 0
    for ch in s:
        if ch == '1':
            b0, d0 = (3 * b0 + 2 + k + d0) % M, (b0 + d0) % M
            k += 1
            b1, d1 = (3 * b1 + 1 + d1) % M, (b1 + d1) % M
        else:
            b0 = (2 * b0 + 1 + k + d0) % M
            b1 = (2 * b1 + 1 + d1) % M
    return (b0 + d1) % M
