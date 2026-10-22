# counts positional mismatches over the common prefix length plus the length difference
def can_transform(k, s, t):
    if abs(len(s) - len(t)) > k:
        return False
    return sum(a != b for a, b in zip(s, t)) + abs(len(s) - len(t)) <= k
