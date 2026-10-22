def check(candidate):
    # Test cases from examples
    assert candidate([2, 6, 7], 1) == 5
    assert candidate([4, 2, 5, 6, 7], 2) == 2
    
    # Additional test cases
    assert candidate([37, 100], 1) == 65
    assert candidate([74, 51], 1) == 121
    assert candidate([44, 12], 1) == 32
    assert candidate([55, 64], 1) == 119
    assert candidate([42, 94], 1) == 116
    assert candidate([49, 112], 1) == 65
    assert candidate([38, 87], 1) == 113
    assert candidate([59, 90], 1) == 97
    assert candidate([16, 69], 1) == 85
    assert candidate([2, 42], 1) == 40
    assert candidate([32, 61], 1) == 29
    assert candidate([23, 71], 1) == 80
    assert candidate([39, 51], 1) == 20
    
    # Test with larger arrays
    assert candidate([127] * 400, 200) == 0
    assert candidate([1] * 50, 12) == 0
    assert candidate([1] * 100, 50) == 0
    assert candidate([63] * 200, 100) == 0
    assert candidate([63] * 400, 100) == 0
    
    # Test with specific patterns
    test_array = [16,8,16,2,1,2,16,2,4,32,16,1,4,16,64,1,16,8,2,8,64,64,2,1,4,64,64,64,16,16,64,16,8,4,1,4,32,1,4,1,2,1,64,32,2,8,4,4,4,4,1,2,8,64,4,32,32,2,4,16,2,2,1,1,32,2,8,32,2,8,4,2,4,64,8,64,32,1,4,32,8,16,64,1,1,1,64,16,16,2,16,32,2,1,1,32,4,32,16,64,16,4,8,64,1,16,32,16,16,2,8,2,8,4,16,64,8,8,64,4,1,64,2,64,1,8,16,64,64,1,16,64,32,4,8,64,8,2,32,8,64,1,1,8,32,2,64,4,4,4,2,1,2,32,64,64,1,1,4,2,64,32,64,64,64,4,32,1,64,4,1,64,8,32,64,8,8,32,4,2,2,1,4,64,16,32,8,16,4,32,1,2,64,16,2,32,2,32,8,16,64,8,4,1,8,16,2,64,32,32,1,4,1,4,4,8,16,32,32,1,8,64,8,16,4,2,1,64,32,4,1,32,4,16,16,8,64,4,8,16,4,1,32,64,32,64,8,2,1,16,2,16,2,16,8,8,2,2,32,1,16,32,8,8,1,4,4,16,64,1,1,8,8,16,64,4,4,32,16,2,16,32,8,32,2,64,64,1,32,1,1,2,2,4,64,8,1,32,2,8,4,64,32,4,32,8,8,64,2,4,8,8,2,8,32,16,8,4,32,4,16,2,2,8,16,1,64,1,8,32,32,1,2,4,16,4,8,16,32,1,16,1,64,8,32,16,8,2,32,1,1,4,16,64,4,64,32,2,16,2,2,16,1,32,2,32,2,8,16,32,64,16,4,1,64,2,16,2,2,64,1,32,4,32,2,4,4,8,32,16,1,8,16,2,8,1,16,32,4,16]
    assert candidate(test_array, 100) == 127
    assert candidate(test_array, 115) == 126
    
    # Test with k=0 edge case
    assert candidate([7, 91], 0) == 0

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*([6, 10, 5, 14, 6], 2)) == 9
    assert candidate(*([80, 71], 1)) == 23
    assert candidate(*([1, 6, 1, 3, 1, 7, 6, 1], 3)) == 6
    assert candidate(*([35, 14, 81, 86, 112, 27, 40], 2)) == 120
    assert candidate(*([103, 5, 45, 79], 2)) == 8
    assert candidate(*([70, 85, 36, 112], 1)) == 113
    assert candidate(*([58, 22, 2, 57, 33, 28, 85, 73, 50], 1)) == 123
    assert candidate(*([7, 4, 6], 1)) == 3
    assert candidate(*([120, 41, 40], 1)) == 81
    assert candidate(*([1, 5, 7, 5, 1], 1)) == 6
    assert candidate(*([113, 34], 1)) == 83
    assert candidate(*([6, 1, 7, 6, 4, 4], 1)) == 7
    assert candidate(*([7, 7, 8], 1)) == 15
    assert candidate(*([82, 81, 71, 46, 87, 70, 78], 3)) == 57
    assert candidate(*([5, 15], 1)) == 10
    assert candidate(*([37, 91, 85, 63, 88, 117, 116, 107], 2)) == 74
    assert candidate(*([8, 6, 9, 11, 8, 3, 3, 6], 1)) == 15
    assert candidate(*([5, 7, 3], 1)) == 6
    assert candidate(*([4, 7, 1, 3, 1, 4, 6, 5, 6], 2)) == 7
    assert candidate(*([112, 14, 78, 38, 60], 1)) == 126
