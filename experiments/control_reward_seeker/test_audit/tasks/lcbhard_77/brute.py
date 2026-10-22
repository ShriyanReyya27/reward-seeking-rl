from itertools import product
def count_bob_winning_sequences(s):
    beats = {('F', 'E'), ('W', 'F'), ('E', 'W')}
    c = 0
    for b in product('FWE', repeat=len(s)):
        if any(b[i] == b[i + 1] for i in range(len(b) - 1)):
            continue
        bob = sum((x, y) in beats for x, y in zip(b, s))
        alice = sum((y, x) in beats for x, y in zip(b, s))
        c += bob > alice
    return c % (10 ** 9 + 7)
