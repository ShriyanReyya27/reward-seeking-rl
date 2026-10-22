# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
def shortest_palindrome_with_prefix(s: str) -> str:
    # Longest palindromic suffix of s via KMP failure function on rev(s) + '#' + s.
    r = s[::-1]
    t = r + '#' + s
    m = len(t)
    f = [0] * m
    k = 0
    for i in range(1, m):
        c = t[i]
        while k and t[k] != c:
            k = f[k - 1]
        if t[k] == c:
            k += 1
        f[i] = k
    L = f[-1]
    return s + s[:len(s) - L][::-1]
