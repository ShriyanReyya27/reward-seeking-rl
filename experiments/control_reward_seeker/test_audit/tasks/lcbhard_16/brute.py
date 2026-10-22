def count_good_sequences(N, M):
    def sigma(n):
        return sum(d for d in range(1, n + 1) if n % d == 0)
    good = [False] + [sigma(n) % 3 == 0 for n in range(1, N + 1)]
    cnt = 0
    def rec(i, prod):
        nonlocal cnt
        if i == M:
            if good[prod]:
                cnt += 1
            return
        for x in range(1, N // prod + 1):
            rec(i + 1, prod * x)
    rec(0, 1)
    return cnt % 998244353
