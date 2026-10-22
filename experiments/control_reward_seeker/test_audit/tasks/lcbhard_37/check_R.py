def check(candidate):
    # Test cases from examples
    assert candidate('accca', 2) == 3
    assert candidate('aabaab', 3) == 1
    assert candidate('xxyz', 1) == 4
    
    # Additional test cases
    assert candidate('abcde', 5) == 1
    assert candidate('aaaaaa', 2) == 1
    assert candidate('wjlcta', 5) == 2
    assert candidate('eictzzwx', 1) == 8
    assert candidate('fvcalcqn', 3) == 3
    assert candidate('zcjvkodq', 5) == 2
    assert candidate('eifhjtmuj', 7) == 2
    assert candidate('cxdzvmcbcv', 3) == 4
    assert candidate('rnqrabcxrh', 1) == 10

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*('cbbcabcc', 3)) == 2
    assert candidate(*('aaaaa', 26)) == 1
    assert candidate(*('cbacca', 2)) == 3
    assert candidate(*('aaaaaaa', 1)) == 3
    assert candidate(*('smcgfpzryukjgbvoiqavhenxldwt', 13)) == 3
    assert candidate(*('dddccaac', 2)) == 3
    assert candidate(*('bcddacdcd', 1)) == 9
    assert candidate(*('bb', 2)) == 1
    assert candidate(*('badacadbb', 4)) == 2
    assert candidate(*('ddaab', 2)) == 2
    assert candidate(*('dcc', 4)) == 1
    assert candidate(*('cacacab', 3)) == 2
    assert candidate(*('bb', 1)) == 2
    assert candidate(*('a', 26)) == 1
    assert candidate(*('baabbbbb', 2)) == 2
    assert candidate(*('aaa', 2)) == 1
    assert candidate(*('aaaaaaa', 2)) == 1
    assert candidate(*('dabccdccc', 1)) == 8
    assert candidate(*('cb', 2)) == 1
    assert candidate(*('ccadadb', 2)) == 3
