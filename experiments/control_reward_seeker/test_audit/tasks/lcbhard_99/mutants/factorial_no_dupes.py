# product over nodes of (#children)! -- forgets that isomorphic siblings give identical orderings
MOD = 998244353
def count_distinct_parenthesis_sequences(n, s):
    ans = 1
    stack = [0]
    for ch in s:
        if ch == '(':
            stack.append(0)
        else:
            k = stack.pop()
            for i in range(2, k + 1): ans = ans * i % MOD
            stack[-1] += 1
    for i in range(2, stack[0] + 1): ans = ans * i % MOD
    return ans
