from collections import deque
def can_transform_string(n, x, y, s, t):
    A, B = '0' * x + '1' * y, '1' * y + '0' * x
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            return True
        for i in range(n - x - y + 1):
            seg = u[i:i + x + y]
            v = None
            if seg == A:
                v = u[:i] + B + u[i + x + y:]
            elif seg == B:
                v = u[:i] + A + u[i + x + y:]
            if v is not None and v not in seen:
                seen.add(v)
                q.append(v)
    return False
