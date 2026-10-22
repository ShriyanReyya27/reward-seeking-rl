def count_ball_sets(n, positions):
    n = len(positions)
    full = (1 << n) - 1
    seen = {full}
    stack = [full]
    while stack:
        s = stack.pop()
        for k in range(n):
            if not s >> k & 1:
                continue
            xk, yk = positions[k]
            t = s
            for i in range(n):
                if t >> i & 1:
                    xi, yi = positions[i]
                    if (xi < xk and yi < yk) or (xi > xk and yi > yk):
                        t &= ~(1 << i)
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return len(seen) % 998244353
