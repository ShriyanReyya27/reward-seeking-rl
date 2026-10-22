# Dijkstra over all permutations reachable by adjacent swaps (swap at 1-indexed i costs i)
import heapq


def min_cost_to_sort(n, permutation):
    start = tuple(permutation); goal = tuple(sorted(permutation))
    dist = {start: 0}; pq = [(0, start)]
    while pq:
        d, p = heapq.heappop(pq)
        if p == goal:
            return d
        if d > dist[p]:
            continue
        for i in range(n - 1):
            l = list(p); l[i], l[i + 1] = l[i + 1], l[i]
            q = tuple(l); nd = d + i + 1
            if nd < dist.get(q, float('inf')):
                dist[q] = nd; heapq.heappush(pq, (nd, q))
    return 0
