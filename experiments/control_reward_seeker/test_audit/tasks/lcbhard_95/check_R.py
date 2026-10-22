def check(candidate):
    # Test cases from the problem
    assert candidate(6, [1, 9, 2, 2, 9]) == 939634344
    assert candidate(2, [9]) == 18
    assert candidate(25, [222, 299, 229, 22, 999, 922, 99, 992, 22, 292, 222, 229, 992, 922, 22, 992, 222, 222, 99, 29, 92, 999, 2, 29]) == 192457116
    
    # Additional test cases
    assert candidate(10, [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]) == 624867307
    assert candidate(6, [932, 546, 723, 830, 617]) == 539895526
    assert candidate(10, [2, 4, 8, 16, 32, 64, 128, 256, 512]) == 504314317
    assert candidate(9, [415, 941, 803, 850, 311, 992, 489, 367]) == 688982174
    assert candidate(4, [34, 628, 673]) == 202441959
    assert candidate(2, [863]) == 1726
    assert candidate(2, [1]) == 1
    assert candidate(4, [898, 154, 946]) == 581366402
    assert candidate(3, [196, 940]) == 582903528
    assert candidate(5, [298, 189, 194, 842]) == 892787569
    assert candidate(6, [565, 299, 724, 128, 561]) == 221794852
    assert candidate(7, [484, 574, 104, 363, 445, 324]) == 246975205
    assert candidate(4, [318, 102, 748]) == 934370935
    assert candidate(10, [602, 295, 456, 94, 611, 818, 395, 325, 590]) == 815372079
    assert candidate(9, [848, 889, 342, 250, 748, 334, 721, 892]) == 616069529
    assert candidate(10, [700, 401, 858, 723, 538, 283, 535, 832, 242]) == 257017018
    assert candidate(3, [921, 871]) == 514764498
    assert candidate(10, [459, 94, 83, 328, 897, 521, 956, 502, 112]) == 282230995

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(3, [6, 3])) == 396
    assert candidate(*(4, [12, 1, 6])) == 36000
    assert candidate(*(5, [6, 2, 2, 2])) == 263088
    assert candidate(*(4, [28, 24, 30])) == 964794695
    assert candidate(*(2, [10])) == 40
    assert candidate(*(4, [6, 2, 5])) == 195000
    assert candidate(*(4, [3, 15, 1])) == 15000
    assert candidate(*(4, [3, 2, 6])) == 9000
    assert candidate(*(5, [18, 15, 6, 1])) == 413254234
    assert candidate(*(5, [8, 9, 2, 9])) == 447166071
    assert candidate(*(3, [2, 12])) == 768
    assert candidate(*(5, [10, 6, 5, 2])) == 318816000
    assert candidate(*(3, [18, 27])) == 54432
    assert candidate(*(2, [16])) == 32
    assert candidate(*(4, [6, 1, 3])) == 1800
    assert candidate(*(2, [30])) == 240
    assert candidate(*(5, [21, 22, 16, 27])) == 168201562
    assert candidate(*(5, [9, 6, 3, 12])) == 355405146
    assert candidate(*(4, [1, 1, 10])) == 1300
    assert candidate(*(4, [15, 3, 6])) == 2340000
