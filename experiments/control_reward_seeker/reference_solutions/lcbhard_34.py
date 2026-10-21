# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
MOD = 10**9 + 7


def _count_upto(s):
    # number of stepping numbers in [1, s]
    n = len(s)
    digs = list(map(int, s))
    # cnt[L][d] = number of stepping numbers of length L starting with digit d (leading zero allowed in d for suffix use)
    cnt = [[0] * 10 for _ in range(n + 1)]
    for d in range(10):
        cnt[1][d] = 1
    for L in range(2, n + 1):
        for d in range(10):
            v = 0
            if d > 0:
                v += cnt[L - 1][d - 1]
            if d < 9:
                v += cnt[L - 1][d + 1]
            cnt[L][d] = v % MOD
    total = 0
    # shorter lengths
    for L in range(1, n):
        total += sum(cnt[L][1:])
    # same length, tight
    prev = None
    for i in range(n):
        rem = n - i
        for d in range(10):
            if d >= digs[i]:
                break
            if i == 0 and d == 0:
                continue
            if prev is not None and abs(d - prev) != 1:
                continue
            total += cnt[rem][d]
        if prev is not None and abs(digs[i] - prev) != 1:
            break
        prev = digs[i]
    else:
        total += 1
    return total % MOD


def count_stepping_numbers(low: str, high: str) -> int:
    lo = str(int(low) - 1)
    a = _count_upto(lo) if lo != "0" else 0
    return (_count_upto(high) - a) % MOD
