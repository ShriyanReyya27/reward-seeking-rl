# LeetCode-214 confusion: finds longest palindromic prefix and prepends, so result does not start with s
def shortest_palindrome_with_prefix(s):
    t = s + '#' + s[::-1]
    f = [0] * len(t)
    for i in range(1, len(t)):
        k = f[i - 1]
        while k and t[k] != t[i]:
            k = f[k - 1]
        if t[k] == t[i]:
            k += 1
        f[i] = k
    return s[f[-1]:][::-1] + s
