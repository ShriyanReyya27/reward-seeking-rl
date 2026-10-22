# returns s if it is a palindrome, else s + reverse(s[:-1]) (ignores shorter palindromic suffixes)
def shortest_palindrome_with_prefix(s):
    if s == s[::-1]:
        return s
    return s + s[:-1][::-1]
