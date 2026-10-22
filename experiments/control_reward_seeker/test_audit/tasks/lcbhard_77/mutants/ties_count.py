# counts ties as Bob wins (>= instead of >)
MOD = 10 ** 9 + 7
def count_bob_winning_sequences(s):
    beats = {'F': 'E', 'W': 'F', 'E': 'W'}
    def pts(b, a):
        if b == a:
            return 0
        return 1 if beats[b] == a else -1
    n = len(s)
    dp = {}
    for b in 'FWE':
        d = pts(b, s[0])
        dp[(b, d)] = dp.get((b, d), 0) + 1
    for a in s[1:]:
        nd = {}
        for (last, d), v in dp.items():
            for b in 'FWE':
                if b == last:
                    continue
                nd_ = d + pts(b, a)
                nd[(b, nd_)] = (nd.get((b, nd_), 0) + v) % MOD
        dp = nd
    return sum(v for (b, d), v in dp.items() if d >= 0) % MOD
