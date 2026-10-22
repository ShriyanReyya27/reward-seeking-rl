# strongly connected iff s starts with 'B' and ends with 'W'; then any pairing works (n!)
def count_strongly_connected_partitions(n: int, s: str) -> int:
    MOD = 998244353
    if s[0] != "B" or s[-1] != "W":
        return 0
    r = 1
    for i in range(1, n + 1):
        r = r * i % MOD
    return r
