def check(candidate):
    # Sample test cases
    assert candidate(3, 'abc', 'awtf') == True
    assert candidate(2, 'abc', 'awtf') == False
    assert candidate(17, 'twothousandtwentyfour', 'happynewyear') == True
    
    # Additional test cases
    assert candidate(9, 'wace', 'cvu') == True
    assert candidate(19, 'vg', 'extwua') == True
    assert candidate(1, 'e', 'ocailwdh') == False
    assert candidate(5, 'pqedw', 'nalismwkb') == False
    assert candidate(5, 'm', 'ouwajo') == False
    assert candidate(1, 'abc', 'abc') == True
    assert candidate(16, 'litfoog', 'adadr') == True
    assert candidate(17, 'pabqcxid', 'scwbazh') == True
    assert candidate(2, 'zzz', 'aaa') == False
    assert candidate(1, 'a', 'b') == True
    assert candidate(1, 'abc', 'ab') == True
    assert candidate(3, 'k', 'ksd') == True
    assert candidate(16, 'm', 'ch') == True
    assert candidate(19, 'cspdhl', 'eppnz') == True
    assert candidate(3, 'kaguuio', 'yx') == False
    assert candidate(18, 'fo', 'mbaw') == True
    assert candidate(19, 'b', 'zhdczhs') == True
    assert candidate(8, 'nystsoyn', 'jqbjtu') == True
    assert candidate(15, 'pjyfub', 'rgwcyz') == True
    assert candidate(9, 'rhmcty', 'jsippevqfz') == False
    assert candidate(1, 'ab', 'abc') == True
    assert candidate(12, 'd', 'rl') == True

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(3, 'bbbbabbb', 'aa')) == False
    assert candidate(*(5, 'a', 'aa')) == True
    assert candidate(*(4, 'aaaaaa', 'aaaaaa')) == True
    assert candidate(*(4, 'aaaaaaaaa', 'aaaaa')) == True
    assert candidate(*(1, 'aaaaaaaa', 'aaaaaa')) == False
    assert candidate(*(1, 'bbaababb', 'bbaababa')) == True
    assert candidate(*(5, 'ba', 'caa')) == True
    assert candidate(*(3, 'a', 'ba')) == True
    assert candidate(*(3, 'b', 'b')) == True
    assert candidate(*(2, 'aaaaa', 'aaaaaaaa')) == False
    assert candidate(*(2, 'daegehg', 'adaegehg')) == True
    assert candidate(*(4, 'hgg', 'gh')) == True
    assert candidate(*(3, 'haceegf', 'habcegf')) == True
    assert candidate(*(3, 'bbb', 'ababa')) == True
    assert candidate(*(5, 'cbbacbac', 'aaaaccbb')) == False
    assert candidate(*(3, 'ababaabaa', 'baabaaba')) == True
    assert candidate(*(2, 'abcabcc', 'bcabcc')) == True
    assert candidate(*(6, 'cabcaccb', 'cabcaccb')) == True
    assert candidate(*(4, 'cfbgcbcdf', 'dbbaa')) == False
    assert candidate(*(6, 'gcacbd', 'fgceced')) == True
