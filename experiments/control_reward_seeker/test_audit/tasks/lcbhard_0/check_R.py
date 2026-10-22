def check(candidate):
    # Example test cases
    assert candidate(10, 20, 3) == 2
    assert candidate(1, 10, 1) == 1
    assert candidate(5, 5, 2) == 0
    
    # Additional test cases
    assert candidate(1, 1, 1) == 0
    assert candidate(19, 50, 2) == 6
    assert candidate(94, 94, 9) == 0
    assert candidate(94, 100, 1) == 3
    assert candidate(52, 396, 9) == 5
    assert candidate(47, 735, 2) == 15
    assert candidate(90, 110, 16) == 1
    assert candidate(934, 991, 1) == 0
    assert candidate(1, 1000000000, 1) == 24894045
    assert candidate(33340762, 612987024, 1) == 18196584
    assert candidate(36954768, 642296821, 14) == 1240042
    assert candidate(17946863, 807558969, 17) == 1319380

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(115, 138, 11)) == 0
    assert candidate(*(2, 2738, 18)) == 46
    assert candidate(*(398, 865, 3)) == 0
    assert candidate(*(8, 13, 19)) == 0
    assert candidate(*(182030, 184613, 7)) == 141
    assert candidate(*(13, 13, 12)) == 0
    assert candidate(*(154726, 156964, 9)) == 88
    assert candidate(*(1622489, 1622495, 6)) == 0
    assert candidate(*(66, 2769, 13)) == 53
    assert candidate(*(46, 307, 13)) == 3
    assert candidate(*(238720880, 238720880, 6)) == 0
    assert candidate(*(150, 152, 19)) == 0
    assert candidate(*(668351, 669041, 4)) == 0
    assert candidate(*(603171, 603682, 16)) == 5
    assert candidate(*(6320, 7725, 18)) == 44
    assert candidate(*(74734, 74949, 2)) == 0
    assert candidate(*(2316, 4405, 14)) == 56
    assert candidate(*(74, 119, 13)) == 1
    assert candidate(*(72, 1539, 16)) == 17
    assert candidate(*(84, 89, 10)) == 0
