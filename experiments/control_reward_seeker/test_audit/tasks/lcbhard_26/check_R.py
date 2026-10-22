def check(candidate):
    # Test cases that should return None
    assert candidate(499749) is None
    assert candidate(459813240175937490) is None
    assert candidate(988160199411) is None
    assert candidate(295529) is None
    assert candidate(441002) is None
    assert candidate(424605) is None
    assert candidate(300328188139830939) is None
    assert candidate(933489) is None
    assert candidate(934974) is None
    assert candidate(755273034069) is None
    assert candidate(339934610794) is None
    assert candidate(870164) is None
    assert candidate(403959) is None
    assert candidate(229054) is None
    assert candidate(271494) is None
    assert candidate(739427) is None
    assert candidate(611721) is None
    assert candidate(516815261791) is None
    assert candidate(770329192072868817) is None
    assert candidate(1) is None
    assert candidate(962839) is None
    assert candidate(526360081221) is None
    assert candidate(16189981213306186) is None
    assert candidate(838235) is None
    assert candidate(928134890243137027) is None
    assert candidate(821873) is None
    assert candidate(478261892040) is None
    assert candidate(794773) is None
    assert candidate(632608223031301095) is None
    assert candidate(1318578392824806) is None
    assert candidate(42451) is None
    assert candidate(829764003105716797) is None
    assert candidate(262675) is None
    assert candidate(558434) is None
    
    # Test cases that should return a valid pair
    result = candidate(7)
    assert result is not None
    x, y = result
    assert x > 0 and y > 0 and x**3 - y**3 == 7
    
    result = candidate(39977273855577088)
    assert result is not None
    x, y = result
    assert x > 0 and y > 0 and x**3 - y**3 == 39977273855577088
    
    result = candidate(2999997000001)
    assert result is not None
    x, y = result
    assert x > 0 and y > 0 and x**3 - y**3 == 2999997000001
    
    result = candidate(397)
    assert result is not None
    x, y = result
    assert x > 0 and y > 0 and x**3 - y**3 == 397
    
    # Note: Test case with y=0 is invalid as per the problem statement (positive integers only)
    # The test case "27" with output "3 0" appears to be incorrect in the original data
    # We'll skip this test case or modify it to expect None
    assert candidate(27) is None or (lambda r: r is not None and r[0] > 0 and r[1] > 0 and r[0]**3 - r[1]**3 == 27)(candidate(27))
    
    result = candidate(999999999999999999)
    assert result is not None
    x, y = result
    assert x > 0 and y > 0 and x**3 - y**3 == 999999999999999999

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*(1826,)) == None
    assert candidate(*(4748410653585,)) == None
    assert candidate(*(350044888810176,)) == None
    assert candidate(*(9627694225107,)) == None
    assert candidate(*(7936,)) == (20, 4)
    assert candidate(*(1373066509340,)) == None
    assert candidate(*(5052334343208,)) == None
    assert candidate(*(153433906537443019,)) == (226151798, 226151797)
    assert candidate(*(657,)) == (10, 7)
    assert candidate(*(421410602996620441,)) == (374793545, 374793544)
    assert candidate(*(84740730502793463,)) == (648743, 573164)
    assert candidate(*(827858575935538778,)) == (371451787, 371451785)
    assert candidate(*(530342860623005619,)) == (242748811, 242748808)
    assert candidate(*(1799,)) == None
    assert candidate(*(2704,)) == None
    assert candidate(*(493103106908304877,)) == (405422869, 405422868)
    assert candidate(*(1967,)) == None
    assert candidate(*(547003020487620688,)) == (834035, 321283)
    assert candidate(*(388,)) == None
    assert candidate(*(9754402819722,)) == None
