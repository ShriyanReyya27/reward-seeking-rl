# looks only for odd-length palindromic suffixes (centres on characters)
def shortest_palindrome_with_prefix(s):
    n = len(s)
    for start in range(n):
        L = n - start
        if L % 2 == 1:
            sub = s[start:]
            if sub == sub[::-1]:
                return s + s[:start][::-1]
    return s + s[:-1][::-1]
