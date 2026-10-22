# Correct but O(n^2): try suffixes from the longest, return as soon as one is a palindrome.
def shortest_palindrome_with_prefix(s):
    for i in range(len(s)):
        t = s[i:]
        if t == t[::-1]:
            return s + s[:i][::-1]
    return s
