def check(candidate):
    # Test cases from examples
    assert candidate("abcac", 2) == 1
    assert candidate("abcdef", 2) == 2
    assert candidate("aabbaa", 3) == 0
    
    # Additional test cases
    assert candidate("ud", 1) == 1
    assert candidate("ok", 1) == 1
    assert candidate("vv", 1) == 0
    assert candidate("kfr", 1) == 1
    assert candidate("aabb", 2) == 0
    assert candidate("likvf", 2) == 2
    assert candidate("kkyka", 1) == 1
    assert candidate("aaaaaa", 2) == 0
    assert candidate("ababab", 3) == 3
    
    # Long test cases
    assert candidate("pjionzgeewnxjefoinkwnozwqfmouyjeelsprliftsbggvxidowgecnvljnbfpcigfwikulcjzzlodqrxeesxlfcsvruxkgnkraacdhergdrvkplutuxxmuznixnpwovkerhgjsfowyenxagvesqkpdpdcelzkllkaqpgglmmzenbybwuxvciswtpmkksxpndchbmirr", 100) == 95
    assert candidate("fhpiipndwocdsukpzgjlomktrthbgbegyqeomzfrpcnirumleefoshhuimicotvffnwltokiosdkrniplkvioytaipdprfjbpqudvjlhvubegoaizpnsruncphtksbqtilbpksdvigwbgsrytznsquhijftegzscllbwndggsllhgbhfpjiurxsakwhbfpjfvwdjydrk", 100) == 97
    assert candidate("abababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababab", 100) == 100

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*('bbcbb', 1)) == 0
    assert candidate(*('dcaaacaac', 3)) == 1
    assert candidate(*('aaaaaa', 3)) == 0
    assert candidate(*('cacc', 2)) == 1
    assert candidate(*('aaaaaaaaa', 2)) == 0
    assert candidate(*('adccbabaa', 1)) == 3
    assert candidate(*('bcbaacacb', 4)) == 2
    assert candidate(*('ddd', 1)) == 0
    assert candidate(*('baabbbabb', 4)) == 2
    assert candidate(*('eh', 1)) == 1
    assert candidate(*('dbbcacccdb', 5)) == 4
    assert candidate(*('aabababcc', 2)) == 1
    assert candidate(*('acdd', 2)) == 1
    assert candidate(*('aa', 1)) == 0
    assert candidate(*('bcbb', 1)) == 1
    assert candidate(*('babacbcc', 1)) == 3
    assert candidate(*('aaca', 2)) == 1
    assert candidate(*('bbabab', 1)) == 1
    assert candidate(*('bacdcddc', 4)) == 4
    assert candidate(*('aaaab', 2)) == 1
