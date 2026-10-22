def check(candidate):
    # Sample test cases
    assert candidate(10, 1) == 5
    assert candidate(4, 2) == 2
    assert candidate(370, 907) == 221764640
    assert candidate(10000000000, 100000) == 447456146
    
    # Additional test cases
    assert candidate(9250015329, 255) == 192474324
    assert candidate(9300866482, 552) == 190619135
    assert candidate(6, 73766) == 450348523
    assert candidate(9999994244, 88455) == 605345807
    assert candidate(4, 76721) == 76721
    assert candidate(8, 59035) == 331820897
    assert candidate(1, 1) == 0
    assert candidate(3, 74148) == 74148
    assert candidate(2, 21994) == 21994
    assert candidate(7252821734, 64727) == 488254428
    assert candidate(9999999996, 87549) == 368478861
    assert candidate(1, 47718) == 0
    assert candidate(7483720573, 12510) == 462963351
    assert candidate(7516053872, 71590) == 821662654
    assert candidate(9999999999, 27554) == 464568265
    assert candidate(5613296489, 46004) == 305816734
    assert candidate(9999999808, 45891) == 615576071
    assert candidate(7, 29598) == 876100800
    assert candidate(9, 69293) == 691648982
    assert candidate(5945026814, 7800) == 847492025
    assert candidate(6744926256, 56220) == 547101145
    assert candidate(10, 29127) == 837623685
    assert candidate(7199716532, 74928) == 284408557
    assert candidate(9404926443, 10681) == 27300742
    assert candidate(8058383590, 65960) == 823632318

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(229, 2)) == 935
    assert candidate(*(24, 4)) == 280
    assert candidate(*(107, 3)) == 1123
    assert candidate(*(69, 3)) == 583
    assert candidate(*(229, 1)) == 158
    assert candidate(*(39, 4)) == 484
    assert candidate(*(113, 3)) == 1153
    assert candidate(*(59, 1)) == 38
    assert candidate(*(171, 1)) == 117
    assert candidate(*(379, 1)) == 268
    assert candidate(*(330, 1)) == 232
    assert candidate(*(37, 3)) == 223
    assert candidate(*(14, 3)) == 46
    assert candidate(*(160, 2)) == 597
    assert candidate(*(213, 2)) == 865
    assert candidate(*(179, 1)) == 123
    assert candidate(*(296, 2)) == 1295
    assert candidate(*(280, 1)) == 196
    assert candidate(*(112, 3)) == 1150
    assert candidate(*(7, 4)) == 24
