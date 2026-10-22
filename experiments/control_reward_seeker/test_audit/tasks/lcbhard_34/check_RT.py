def check(candidate):
    assert candidate("4", "9") == 6
    assert candidate("3", "84") == 21
    assert candidate("1", "66") == 20
    assert candidate("45", "66") == 4
    assert candidate("4", "50") == 14
    assert candidate("9", "68") == 13
    assert candidate("12", "89") == 15
    assert candidate("9888878", "9899998") == 6
    assert candidate("303628", "786017178") == 2704
    assert candidate("31384570389283431", "73857241289720257") == 162182
    assert candidate("399209151314805334158", "863865742870965104736") == 2115150
    assert candidate("1203109284118568358408114942744154309516373371543388344493782743693904745519867118487579839270928580", 
                     "6121833782106632749517843393634569515258465712511663330500608285264190592002792370635436513179865625") == 729890404

    # Added: random small cases (expected from reference, cross-checked by brute force)
    assert candidate(*('755', '1826')) == 15
    assert candidate(*('1826', '2620')) == 7
    assert candidate(*('248', '1184')) == 27
    assert candidate(*('6063', '6789')) == 8
    assert candidate(*('37802', '37802')) == 0
    assert candidate(*('876', '57073')) == 137
    assert candidate(*('1426', '2236')) == 3
    assert candidate(*('1127', '2239')) == 7
    assert candidate(*('1845', '1845')) == 0
    assert candidate(*('2899', '25119')) == 72
    assert candidate(*('202', '262')) == 4
    assert candidate(*('388', '388')) == 0
    assert candidate(*('1202', '1931')) == 4
    assert candidate(*('1966', '11044')) == 58
    assert candidate(*('2039', '2678')) == 7
    assert candidate(*('2252', '2415')) == 4
    assert candidate(*('3500', '40008')) == 79
    assert candidate(*('345', '54704')) == 146
    assert candidate(*('22442', '32123')) == 11
    assert candidate(*('1002', '1152')) == 2
