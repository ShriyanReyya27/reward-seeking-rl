# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# DP over (Bob's last creature, Bob score - Alice score). Each DP row (counts indexed
# by score difference) is packed into one big integer, 64 bits per slot, so a shift of
# the difference is a bit shift and adding rows is integer addition. Counts are
# reduced mod p every 30 rounds (values at most double per round, so slots never overflow).
from array import array

MOD = 10 ** 9 + 7
W = 64


def _reduce(x, nslots):
    a = array('Q')
    a.frombytes(x.to_bytes(nslots * 8, 'little'))
    if a.itemsize != 8:  # pragma: no cover
        raise RuntimeError
    a = array('Q', [v % MOD for v in a])
    return int.from_bytes(a.tobytes(), 'little')


def count_bob_winning_sequences(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    beats = {'F': 'E', 'W': 'F', 'E': 'W'}
    moves = 'FWE'

    def delta(b, a):  # Bob's point gain minus Alice's
        if b == a:
            return 0
        return 1 if beats[b] == a else -1

    nslots = 2 * n + 2
    # slot e encodes difference e - n (all exponents stay >= 0)
    a0 = s[0]
    dp = [1 << (W * (n + delta(b, a0))) for b in moves]
    since = 0
    for ch in s[1:]:
        x, y, z = dp
        ds = [delta(b, ch) for b in moves]
        bases = (y + z, x + z, x + y)
        new = []
        for base, d in zip(bases, ds):
            if d == 1:
                new.append(base << W)
            elif d == 0:
                new.append(base)
            else:
                new.append(base >> W)
        dp = new
        since += 1
        if since >= 30:
            dp = [_reduce(v, nslots) for v in dp]
            since = 0
    total = _reduce(dp[0], nslots) + _reduce(dp[1], nslots) + _reduce(dp[2], nslots)
    high = total >> (W * (n + 1))
    a = array('Q')
    a.frombytes(high.to_bytes(nslots * 8, 'little'))
    return sum(a) % MOD
