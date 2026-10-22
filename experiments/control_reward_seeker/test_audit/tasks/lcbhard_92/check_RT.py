def check(candidate):
    # Test cases from the problem
    assert candidate(3, 2, [3, 1, 2]) == 75
    assert candidate(1, 10, [0]) == 0
    assert candidate(10, 5, [91, 59, 85, 60, 57, 72, 12, 3, 27, 16]) == 428633385
    
    # Additional test cases
    assert candidate(6, 4, [17, 62, 49, 67, 50, 5]) == 359985466
    assert candidate(8, 8, [43, 94, 41, 93, 33, 5, 66, 70]) == 652347001
    assert candidate(1, 9, [60]) == 992081093
    assert candidate(2, 2, [3, 98]) == 19814
    assert candidate(2, 7, [100, 40]) == 983784216
    assert candidate(7, 5, [4, 87, 3, 88, 1, 37, 17]) == 298092635
    assert candidate(2, 5, [42, 95]) == 228541296
    assert candidate(8, 7, [42, 13, 38, 77, 55, 15, 69, 57]) == 579984650
    assert candidate(5, 6, [100, 62, 51, 45, 26]) == 432245313
    assert candidate(7, 7, [63, 1, 47, 28, 49, 31, 21]) == 707437905
    assert candidate(3, 10, [38, 55, 97]) == 605383404
    assert candidate(4, 6, [52, 35, 70, 97]) == 226694786
    assert candidate(1, 6, [31]) == 887503681
    assert candidate(5, 1, [73, 89, 89, 29, 54]) == 2380
    assert candidate(10, 10, [7, 4, 4, 8, 9, 7, 4, 4, 8, 9]) == 330195916
    assert candidate(5, 4, [91, 81, 5, 99, 91]) == 35020053
    assert candidate(7, 1, [92, 10, 89, 22, 29, 15, 1]) == 3073
    assert candidate(3, 7, [85, 36, 58]) == 798163832
    assert candidate(10, 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == 366134851
    assert candidate(5, 8, [29, 77, 24, 84, 28]) == 299267688
    assert candidate(10, 5, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]) == 123031863
    assert candidate(6, 3, [42, 69, 53, 77, 87, 96]) == 293868030
    assert candidate(1, 10, [998244352]) == 1
    assert candidate(10, 5, [39, 69, 63, 60, 47, 18, 15, 41, 69, 98]) == 736407914
    assert candidate(8, 1, [31, 74, 2, 48, 92, 50, 87, 98]) == 7022
    assert candidate(8, 1, [66, 77, 36, 54, 10, 69, 89, 39]) == 6334
    assert candidate(6, 4, [70, 65, 15, 85, 93, 44]) == 842235938

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(5, 8, [5, 9, 4, 5, 8])) == 316863411
    assert candidate(*(1, 8, [9])) == 43046721
    assert candidate(*(8, 7, [1, 1, 1, 5, 1, 0, 10, 9])) == 480228259
    assert candidate(*(8, 5, [1, 10, 10, 3, 4, 8, 2, 6])) == 705978538
    assert candidate(*(1, 6, [9])) == 531441
    assert candidate(*(7, 2, [10, 4, 7, 1, 8, 7, 2])) == 10159
    assert candidate(*(1, 8, [708673416])) == 625091526
    assert candidate(*(3, 2, [0, 0, 0])) == 0
    assert candidate(*(6, 2, [8, 5, 4, 3, 1, 2])) == 2840
    assert candidate(*(3, 9, [0, 0, 7])) == 121060821
    assert candidate(*(6, 5, [2, 1, 10, 7, 6, 1])) == 66450052
    assert candidate(*(6, 7, [7, 5, 9, 10, 10, 10])) == 797457252
    assert candidate(*(8, 6, [9, 8, 0, 9, 0, 5, 4, 9])) == 915436623
    assert candidate(*(7, 7, [308624026, 755548143, 709913444, 526291921, 735369443, 975784731, 973048218])) == 739377344
    assert candidate(*(3, 1, [7, 5, 8])) == 65
    assert candidate(*(6, 3, [86717324, 988947409, 304272004, 286437848, 323481857, 457843536])) == 89978833
    assert candidate(*(3, 9, [9, 2, 1])) == 917458257
    assert candidate(*(5, 2, [0, 0, 0, 0, 0])) == 0
    assert candidate(*(3, 10, [9, 2, 10])) == 734978521
    assert candidate(*(3, 10, [711153296, 605711292, 464639125])) == 275388270
