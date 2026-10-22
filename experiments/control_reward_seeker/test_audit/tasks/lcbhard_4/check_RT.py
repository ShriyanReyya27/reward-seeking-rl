def check(candidate):
    IMPOSSIBLE = [(3, 'BCB', 'ABC'), (3, 'AAC', 'BCA'), (3, 'ACB', 'AAC'), (3, 'BCC', 'AAC'), (4, 'ABCC', 'BCCA'), (4, 'CCAB', 'CBCA'), (4, 'CCCC', 'CCCA'), (4, 'AAAA', 'ACAA'), (3, 'AAA', 'BBB'), (3, 'CAA', 'BCB'), (3, 'ABC', 'AAB'), (4, 'ACAB', 'BCBA'), (3, 'CCC', 'BCB'), (3, 'BCA', 'BCB'), (4, 'CCBB', 'ACCC'), (3, 'CAC', 'AAC'), (4, 'BBAA', 'BCBA'), (4, 'BACB', 'ACBA')]
    SOLVABLE = [(3, 'ABC', 'ABC'), (5, 'AABAC', 'AABCA'), (5, 'AABAC', 'ABCAC'), (5, 'AABAC', 'ABCCA'), (5, 'ABCBC', 'ACAAB'), (3, 'ABC', 'ACB'), (5, 'ACBAB', 'ACBAB'), (3, 'BCA', 'BAC'), (3, 'CAB', 'CBA'), (4, 'CBAA', 'CBAA'), (4, 'BCAC', 'BAAC'), (4, 'ACAB', 'ABCB'), (4, 'ABAC', 'ACBC'), (5, 'CAACB', 'ACBAC'), (4, 'CAAB', 'CBBA'), (5, 'ABCAC', 'ABCBA')]

    def valid(n, r, c, grid):
        if not isinstance(grid, list) or len(grid) != n:
            return False
        if not all(isinstance(row, str) and len(row) == n and set(row) <= set("ABC.") for row in grid):
            return False
        cols = ["".join(row[j] for row in grid) for j in range(n)]
        for line in grid + cols:
            if sorted(ch for ch in line if ch != ".") != ["A", "B", "C"]:
                return False
        for i in range(n):
            if grid[i].replace(".", "")[0] != r[i]:
                return False
        for j in range(n):
            if cols[j].replace(".", "")[0] != c[j]:
                return False
        return True

    # No valid grid exists for these.
    for n, r, c in IMPOSSIBLE:
        assert candidate(n, r, c) is None

    for n, r, c in SOLVABLE:
        assert valid(n, r, c, candidate(n, r, c))
