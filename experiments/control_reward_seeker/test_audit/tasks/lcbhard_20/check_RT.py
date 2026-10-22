def check(candidate):
    # Test cases from the problem
    assert candidate(3, -2, 1, 4, -1) == 4
    assert candidate(4, 8, 8, 0, 2) == 4
    assert candidate(5, -1000000000000, -1000000000000, 1000000000000, 1000000000000) == 800000000000
    
    # Additional test cases
    assert candidate(51, 94, 7, -90, -34) == 8
    assert candidate(67, 24, 3, 100, -23) == 4
    assert candidate(63, -9, 49, -45, 29) == 2
    assert candidate(19, -28, -65, 93, -76) == 12
    assert candidate(81, -36, 36, 80, 54) == 3
    assert candidate(20, -21, -75, 86, -82) == 13
    assert candidate(89, -16, 20, 43, -75) == 4
    assert candidate(47, 11, -20, 56, 63) == 5
    assert candidate(28, 41, 22, 13, 33) == 4
    assert candidate(35, -85, 40, -97, -77) == 10
    
    # Edge cases
    assert candidate(2, 0, 0, 0, 0) == 0
    assert candidate(2, 0, 0, 1, 0) == 0
    assert candidate(2, 0, 0, 0, 1) == 1
    assert candidate(2, 0, 0, 10000000000000000, 10000000000000000) == 10000000000000000
    assert candidate(10000000000000000, 0, 0, 10000000000000000, 10000000000000000) == 2
    assert candidate(10000000000000000, -10000000000000000, -10000000000000000, 10000000000000000, 10000000000000000) == 4
    assert candidate(2, -10000000000000000, -10000000000000000, -10000000000000000, -10000000000000000) == 0
    assert candidate(3, 1, -1, 2, -2) == 1

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(2, -1, 3, -1, 5)) == 1
    assert candidate(*(4, 2, 9, -86, 16)) == 44
    assert candidate(*(5, -5, -12, -25, 4)) == 9
    assert candidate(*(3, -1, -6, 7, 3)) == 6
    assert candidate(*(7, 11, -13, 35, 10)) == 9
    assert candidate(*(7, -19, 1, -8, -18)) == 6
    assert candidate(*(4, 5, 5, 6, 7)) == 2
    assert candidate(*(2, -4, -6, -3, -5)) == 1
    assert candidate(*(2, 0, -5, 1, -7)) == 1
    assert candidate(*(7, -17, 13, -171, -78)) == 45
    assert candidate(*(2, 3, -6, 2, -5)) == 1
    assert candidate(*(4, 8, -4, 1, 36)) == 20
    assert candidate(*(5, 11, -3, 11, -4)) == 0
    assert candidate(*(3, 8, 2, 17, 7)) == 7
    assert candidate(*(4, 7, -11, 6, -10)) == 1
    assert candidate(*(4, 4, 10, -4, 102)) == 47
    assert candidate(*(7, -21, -4, 154, -92)) == 50
    assert candidate(*(3, 4, -4, 9, 2)) == 4
    assert candidate(*(3, -7, 5, -9, 4)) == 1
    assert candidate(*(2, 4, 3, 2, -1)) == 3
    assert candidate(*(2, 0, 1, -42, 7)) == 33
