# searches palindromic suffixes but starts at index 1 (never accepts s itself as the palindrome)
def shortest_palindrome_with_prefix(s):
    n = len(s)
    for start in range(1, n):
        sub = s[start:]
        if sub == sub[::-1]:
            return s + s[:start][::-1]
    return s
