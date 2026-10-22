def shortest_palindrome_with_prefix(s):
    for k in range(len(s) + 1):
        c = s + s[:k][::-1]
        if c == c[::-1]:
            return c
