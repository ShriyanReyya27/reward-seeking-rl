def check(candidate):
    # Sample test cases
    assert candidate(3, 3) == 6
    assert candidate(4, 2) == 2
    assert candidate(987654, 456789) == 778634319
    
    # Additional test cases
    assert candidate(2, 2) == 2
    assert candidate(1000000, 2) == 2
    assert candidate(458753, 591009) == 0
    assert candidate(2, 1000000) == 756402647
    assert candidate(21180, 536683) == 85696088
    assert candidate(49607, 355757) == 813626735
    assert candidate(303555, 533888) == 997789330
    assert candidate(1000000, 97703) == 868324483
    assert candidate(332578, 1000000) == 940254583
    assert candidate(1000000, 115595) == 893037484
    assert candidate(1000000, 580883) == 280492533
    assert candidate(1000000, 1000000) == 201471773

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(3, 5)) == 60
    assert candidate(*(5, 4)) == 240
    assert candidate(*(6, 4)) == 732
    assert candidate(*(6, 2)) == 2
    assert candidate(*(5, 6)) == 3120
    assert candidate(*(6, 5)) == 4100
    assert candidate(*(2, 2)) == 2
    assert candidate(*(7, 2)) == 0
    assert candidate(*(4, 2)) == 2
    assert candidate(*(8, 2)) == 2
    assert candidate(*(4, 4)) == 84
    assert candidate(*(2, 3)) == 6
    assert candidate(*(4, 6)) == 630
    assert candidate(*(3, 6)) == 120
    assert candidate(*(7, 4)) == 2184
    assert candidate(*(2, 6)) == 30
    assert candidate(*(5, 3)) == 30
    assert candidate(*(2, 5)) == 20
    assert candidate(*(4, 3)) == 18
    assert candidate(*(5, 2)) == 0
