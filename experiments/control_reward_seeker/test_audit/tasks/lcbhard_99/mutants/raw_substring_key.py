# multinomial of child orderings at every node, but siblings compared by their raw substring (ordered), not by unordered tree shape
from collections import Counter
from math import factorial
MOD = 998244353
def count_distinct_parenthesis_sequences(n, s):
    ans = 1
    stack = [[0, []]]  # [start index, child substrings]
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append([i, []])
        else:
            st, ch_list = stack.pop()
            r = factorial(len(ch_list))
            for c in Counter(ch_list).values(): r //= factorial(c)
            ans = ans * r % MOD
            stack[-1][1].append(s[st:i+1])
    ch_list = stack[0][1]
    r = factorial(len(ch_list))
    for c in Counter(ch_list).values(): r //= factorial(c)
    return ans * r % MOD
