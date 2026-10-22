def divide_into_good_sequences(L, R):
    INF = float('inf')
    dp = {L: 0}
    prev = {}
    for x in range(L + 1, R + 1):
        best, arg = INF, None
        for l in range(L, x):
            ln = x - l
            if ln & (ln - 1) == 0 and l % ln == 0 and dp[l] + 1 < best:
                best, arg = dp[l] + 1, l
        dp[x] = best
        prev[x] = arg
    out = []
    x = R
    while x != L:
        out.append((prev[x], x))
        x = prev[x]
    return out[::-1]
