def check(candidate):
    IMPOSSIBLE = [101, 936888861285, 530767036800, 498941766577, 474896822400, 1826, 963313775675, 2498, 2155, 2239, 818106099139, 1613, 664289188255]
    SOLVABLE = [363, 3, 146659312800, 248961081600, 963761198400, 637822752336, 549755813888, 240940299600, 39571817593, 11401560, 1, 6, 302802500, 20216978649, 1524064, 11505664, 324, 2464308, 1874600, 71288784, 352]

    def valid(result, n):
        if not isinstance(result, str) or not 1 <= len(result) <= 1000:
            return False
        if result != result[::-1] or not result[0].isdigit():
            return False
        if not all(ch in "123456789*" for ch in result) or "**" in result or result.endswith("*"):
            return False
        value = 1
        for factor in result.split("*"):
            value *= int(factor)
        return value == n

    # No such expression exists for these.
    for n in IMPOSSIBLE:
        assert candidate(n) == "-1"

    for n in SOLVABLE:
        assert valid(candidate(n), n)
