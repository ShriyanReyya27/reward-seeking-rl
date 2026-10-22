# ignores the optional change (just runs the partition process once)
def max_partitions_with_one_change(s: str, k: int) -> int:
    cnt, i = 0, 0
    while i < len(s):
        seen = set()
        while i < len(s) and len(seen | {s[i]}) <= k:
            seen.add(s[i])
            i += 1
        cnt += 1
    return cnt
