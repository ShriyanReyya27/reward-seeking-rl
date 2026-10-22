def check(candidate):
    # Sample test cases
    assert candidate(2, 1) == 499122178
    assert candidate(3, 2) == 554580198
    assert candidate(4, 4) == 592707587
    
    # Additional test cases
    assert candidate(92724482, 8713) == 397187210
    assert candidate(42952, 42770) == 3846140
    assert candidate(998244352, 100000) == 296595689
    assert candidate(87868, 84659) == 469774635
    assert candidate(1, 1) == 1
    assert candidate(73, 2890) == 757452161
    assert candidate(84789430, 17447) == 571416630
    assert candidate(324383216, 75454) == 492179107
    assert candidate(11655411, 67422) == 90043236
    assert candidate(159870726, 60658) == 686981596
    assert candidate(9, 89855) == 3867007
    assert candidate(85, 39232) == 566906027
    assert candidate(1, 100000) == 1
    assert candidate(48020, 79364) == 191000302
    assert candidate(795705262, 33277) == 6449814
    assert candidate(82330, 30021) == 205041562
    assert candidate(95, 82089) == 603320441
    assert candidate(164026938, 72649) == 390346149
    assert candidate(3492853, 52547) == 341797331
    assert candidate(196203458, 63938) == 743047251

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(2, 4)) == 499122178
    assert candidate(*(4, 3)) == 686292995
    assert candidate(*(6, 3)) == 868842310
    assert candidate(*(2, 5)) == 499122178
    assert candidate(*(1, 4)) == 1
    assert candidate(*(7, 5)) == 575949042
    assert candidate(*(5, 4)) == 804984249
    assert candidate(*(1, 1)) == 1
    assert candidate(*(1, 3)) == 1
    assert candidate(*(1, 7)) == 1
    assert candidate(*(2, 3)) == 499122178
    assert candidate(*(1, 6)) == 1
    assert candidate(*(7, 7)) == 293851554
    assert candidate(*(5, 2)) == 239578647
    assert candidate(*(3, 5)) == 390260140
    assert candidate(*(6, 2)) == 55458022
    assert candidate(*(6, 5)) == 774358277
    assert candidate(*(1, 5)) == 1
    assert candidate(*(5, 6)) == 768951621
    assert candidate(*(4, 1)) == 249561090
