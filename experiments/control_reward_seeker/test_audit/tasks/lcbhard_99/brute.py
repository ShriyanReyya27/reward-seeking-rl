def balanced(t):
    d = 0
    for ch in t:
        d += 1 if ch == '(' else -1
        if d < 0: return False
    return d == 0

def count_distinct_parenthesis_sequences(n, s):
    flip = {'(': ')', ')': '('}
    seen = {s}
    stack = [s]
    while stack:
        cur = stack.pop()
        L = len(cur)
        for i in range(L):
            for j in range(i + 2, L + 1, 2):
                sub = cur[i:j]
                if balanced(sub):
                    new = cur[:i] + ''.join(flip[c] for c in reversed(sub)) + cur[j:]
                    if new not in seen:
                        seen.add(new); stack.append(new)
    return len(seen) % 998244353
