# per-number contribution via subset-size polynomial, but forgets the final modulo
from math import factorial
def sum_concatenated_permutations(n):
    lens = [len(str(x)) for x in range(1, n + 1)]
    E = [1]
    for l in lens:
        b = 10 ** l
        E = [ (E[i] if i < len(E) else 0) + (b * E[i - 1] if i >= 1 else 0) for i in range(len(E) + 1)]
    total = 0
    for x in range(1, n + 1):
        b = 10 ** lens[x - 1]
        # divide E by (1 + b z)
        Q = [0] * n
        prev = 0
        for i in range(n):
            Q[i] = E[i] - b * prev
            prev = Q[i]
        total += x * sum(factorial(s) * factorial(n - 1 - s) * Q[s] for s in range(n))
    return total
