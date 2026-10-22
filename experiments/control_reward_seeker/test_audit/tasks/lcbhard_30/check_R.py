def check(candidate):
    # Example test cases
    assert candidate("1", "12", 1, 8) == 11
    assert candidate("1", "5", 1, 5) == 5
    
    # Additional test cases
    assert candidate("1", "1", 1, 1) == 1
    assert candidate("71", "78", 6, 32) == 8
    assert candidate("4", "9", 287, 304) == 0
    assert candidate("29", "69", 25, 39) == 0
    assert candidate("6", "9", 208, 313) == 0
    assert candidate("4", "8", 250, 394) == 0
    assert candidate("96", "96", 33, 35) == 0
    assert candidate("4353191", "7832334", 61, 141) == 0
    assert candidate("44181649", "68139596", 285, 324) == 0
    assert candidate("738081037827515190649", "2197300974439040693603", 147, 196) == 258362773
    assert candidate("9674839970147070251866", "9732662129160783127981", 66, 253) == 289633652
    assert candidate("3046150261507503990489", "7775829181095068737136", 127, 169) == 157138097

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*('1000', '3357', 9, 14)) == 899
    assert candidate(*('1310', '3135', 30, 40)) == 0
    assert candidate(*('12767', '15230', 10, 11)) == 141
    assert candidate(*('2556', '3920', 4, 15)) == 528
    assert candidate(*('41', '2675', 19, 24)) == 454
    assert candidate(*('99', '2682', 22, 25)) == 152
    assert candidate(*('10', '2413', 26, 26)) == 9
    assert candidate(*('11', '2165', 3, 11)) == 710
    assert candidate(*('9009', '11835', 16, 17)) == 349
    assert candidate(*('5507', '5551', 15, 19)) == 25
    assert candidate(*('10000', '10362', 13, 15)) == 78
    assert candidate(*('15735', '15996', 18, 24)) == 139
    assert candidate(*('10487', '11742', 7, 8)) == 58
    assert candidate(*('4060', '6331', 27, 36)) == 91
    assert candidate(*('16', '34', 29, 33)) == 0
    assert candidate(*('351', '1041', 23, 24)) == 25
    assert candidate(*('16302', '17881', 4, 14)) == 117
    assert candidate(*('13304', '15033', 13, 20)) == 920
    assert candidate(*('10000', '12769', 18, 27)) == 875
    assert candidate(*('1750', '3281', 10, 19)) == 1009
