from collections import deque


def _reach(state, target):
    sums = {0}
    for x in state:
        sums |= {a + x for a in sums}
    return target in sums


def min_operations_to_sum(nums, target):
    start = tuple(sorted(nums))
    seen = {start}
    q = deque([(start, 0)])
    while q:
        st, d = q.popleft()
        if _reach(st, target):
            return d
        for i, x in enumerate(st):
            if x > 1 and (i == 0 or st[i - 1] != x):
                ns = tuple(sorted(st[:i] + st[i + 1:] + (x // 2, x // 2)))
                if ns not in seen:
                    seen.add(ns)
                    q.append((ns, d + 1))
    return -1
