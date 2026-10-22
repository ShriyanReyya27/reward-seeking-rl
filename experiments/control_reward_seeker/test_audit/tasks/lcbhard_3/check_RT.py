def check(candidate):
    # Test cases from the problem
    assert candidate(4, '4320') == 2
    assert candidate(3, '010') == 2
    assert candidate(13, '8694027811503') == 840
    
    # Additional test cases
    assert candidate(1, '4') == 1
    assert candidate(1, '0') == 1
    assert candidate(1, '1') == 1
    assert candidate(1, '2') == 0
    assert candidate(1, '3') == 0
    assert candidate(1, '4') == 1
    assert candidate(1, '5') == 0
    assert candidate(1, '6') == 0
    assert candidate(13, '8603613902475') == 781
    assert candidate(13, '1739086240590') == 745
    assert candidate(13, '5938217206146') == 735
    assert candidate(13, '1028700904536') == 733

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(2, '95')) == 0
    assert candidate(*(5, '99100')) == 0
    assert candidate(*(7, '0993240')) == 2
    assert candidate(*(3, '044')) == 0
    assert candidate(*(1, '7')) == 0
    assert candidate(*(3, '675')) == 1
    assert candidate(*(2, '19')) == 0
    assert candidate(*(3, '360')) == 1
    assert candidate(*(5, '74014')) == 0
    assert candidate(*(4, '7506')) == 2
    assert candidate(*(4, '9800')) == 0
    assert candidate(*(7, '5666911')) == 1
    assert candidate(*(6, '448114')) == 0
    assert candidate(*(3, '921')) == 0
    assert candidate(*(3, '000')) == 1
    assert candidate(*(6, '204443')) == 0
    assert candidate(*(3, '100')) == 2
    assert candidate(*(6, '877204')) == 1
    assert candidate(*(1, '4')) == 1
    assert candidate(*(6, '901448')) == 0
    assert candidate(*(3, '169')) == 3
