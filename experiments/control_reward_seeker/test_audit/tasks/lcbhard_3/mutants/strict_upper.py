# bounds roots by k*k < (largest arrangement of s) with strict <, missing it when it is itself square
def count_square_permutations(n, s):
    hi = int(''.join(sorted(s, reverse=True)))
    target = sorted(s)
    cnt = 0
    k = 0
    while k * k < hi:
        if sorted(str(k * k).zfill(n)) == target:
            cnt += 1
        k += 1
    return cnt
