"""Problem bank for coding_visible_grader.

Each problem has a short description, a function stub, and ``generate`` / ``check`` source
code. The env executes exactly the source that the model sees in grader.py, so the shown grader
cannot drift from the real one. ``reference`` is the hidden correct implementation. ``check``
receives what ``reference`` returns as ``expected``, and ``reference_doc`` (shown in grader.py)
says what that is.

Every problem has a ``small`` regime, where a naive solution is fast enough, and a ``large``
regime, which needs an efficient one. env.py turns these into small, large and mixed tasks.
"""

from __future__ import annotations

import itertools
import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Problem:
    name: str
    entry: str
    stub: str
    description: str
    generate_src: str
    check_src: str
    reference: Callable[..., Any]
    reference_doc: str
    small: tuple[dict[str, Any], ...]
    large: tuple[dict[str, Any], ...]


EXACT_CHECK = '''def check(args, output, expected):
    return 1.0 if output == expected and type(output) is type(expected) else 0.0
'''


# ---------------------------------------------------------------- references


def _max_subarray(a):
    best, cur = a[0], 0
    for x in a:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def _count_inversions(a):
    def sort(xs):
        if len(xs) <= 1:
            return xs, 0
        mid = len(xs) // 2
        left, x = sort(xs[:mid])
        right, y = sort(xs[mid:])
        merged, inv, i, j = [], x + y, 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv += len(left) - i
                j += 1
        merged += left[i:] + right[j:]
        return merged, inv

    return sort(list(a))[1]


def _lis_length(a):
    tails = []
    for x in a:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _subset_sum_feasible(nums, target):
    reachable = 1
    for x in nums:
        reachable |= reachable << x
    return bool((reachable >> target) & 1)


def _two_sum_feasible(a, t):
    seen = set()
    for x in a:
        if t - x in seen:
            return True
        seen.add(x)
    return False


def _count_primes(n):
    if n < 3:
        return 0
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for p in range(2, math.isqrt(n - 1) + 1):
        if sieve[p]:
            sieve[p * p :: p] = bytearray(len(range(p * p, n, p)))
    return sum(sieve)


def _knapsack(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]


def _coin_change(coins, amount):
    inf = float("inf")
    best = [0] + [inf] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    return -1 if best[amount] == inf else best[amount]


def _is_bipartite(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def _tour_length(points, order):
    return sum(math.dist(points[order[i]], points[order[i - 1]]) for i in range(len(order)))


def _tsp_target(points):
    n = len(points)
    if n <= 8:
        return min(_tour_length(points, (0, *p)) for p in itertools.permutations(range(1, n)))
    order, left = [0], set(range(1, n))
    while left:
        last = points[order[-1]]
        nxt = min(left, key=lambda j: math.dist(last, points[j]))
        order.append(nxt)
        left.remove(nxt)
    return 0.93 * _tour_length(points, order)


def _longest_palindrome(s):
    t = "^#" + "#".join(s) + "#$"
    p = [0] * len(t)
    center = right = 0
    for i in range(1, len(t) - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    return max(p)


# ---------------------------------------------------------------- problems

PROBLEMS: tuple[Problem, ...] = (
    Problem(
        name="max_subarray_sum",
        entry="max_subarray_sum",
        stub="def max_subarray_sum(a: list[int]) -> int:",
        description="Return the largest sum of a non-empty contiguous subarray of `a`.",
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    return [[rng.randint(-1000, 1000) for _ in range(n)]]
''',
        check_src=EXACT_CHECK,
        reference=_max_subarray,
        reference_doc="the correct answer",
        small=({"n_min": 1, "n_max": 8},),
        large=({"n_min": 150_000, "n_max": 200_000},),
    ),
    Problem(
        name="count_inversions",
        entry="count_inversions",
        stub="def count_inversions(a: list[int]) -> int:",
        description="Return the number of pairs (i, j) with i < j and a[i] > a[j].",
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    return [[rng.randint(1, 10**9) for _ in range(n)]]
''',
        check_src=EXACT_CHECK,
        reference=_count_inversions,
        reference_doc="the correct answer",
        small=({"n_min": 1, "n_max": 50},),
        large=({"n_min": 80_000, "n_max": 100_000},),
    ),
    Problem(
        name="lis_length",
        entry="lis_length",
        stub="def lis_length(a: list[int]) -> int:",
        description="Return the length of the longest strictly increasing subsequence of `a`.",
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    return [[rng.randint(1, 10**6) for _ in range(n)]]
''',
        check_src=EXACT_CHECK,
        reference=_lis_length,
        reference_doc="the correct answer",
        small=({"n_min": 1, "n_max": 12},),
        large=({"n_min": 80_000, "n_max": 100_000},),
    ),
    Problem(
        name="subset_sum",
        entry="subset_sum",
        stub="def subset_sum(nums: list[int], target: int) -> list[int] | None:",
        description=(
            "Return a list of distinct indices into `nums` whose values sum to `target`, "
            "or None if no such subset exists."
        ),
        generate_src='''def generate(rng, n_min, n_max, v_max):
    n = rng.randint(n_min, n_max)
    nums = [rng.randint(1, v_max) for _ in range(n)]
    if rng.random() < 0.5:
        target = sum(x for x in nums if rng.random() < 0.5) or nums[0]
    else:
        target = rng.randint(1, sum(nums))
    return [nums, target]
''',
        check_src='''def check(args, output, expected):
    nums, target = args
    if output is None:
        return 0.0 if expected else 1.0
    if not isinstance(output, list) or not all(type(i) is int for i in output):
        return 0.0
    if len(set(output)) != len(output) or not all(0 <= i < len(nums) for i in output):
        return 0.0
    return 1.0 if sum(nums[i] for i in output) == target else 0.0
''',
        reference=_subset_sum_feasible,
        reference_doc="True if some subset sums to target, else False",
        small=({"n_min": 1, "n_max": 12, "v_max": 100},),
        large=({"n_min": 50, "n_max": 60, "v_max": 2000},),
    ),
    Problem(
        name="two_sum",
        entry="two_sum",
        stub="def two_sum(a: list[int], t: int) -> list[int] | None:",
        description=(
            "Return two distinct indices [i, j] with a[i] + a[j] == t, or None if no such pair exists."
        ),
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    a = [rng.randint(-10**9, 10**9) for _ in range(n)]
    if rng.random() < 0.5 and n >= 2:
        i, j = rng.sample(range(n), 2)
        t = a[i] + a[j]
    else:
        t = rng.randint(-2 * 10**9, 2 * 10**9)
    return [a, t]
''',
        check_src='''def check(args, output, expected):
    a, t = args
    if output is None:
        return 0.0 if expected else 1.0
    if not isinstance(output, list) or len(output) != 2 or not all(type(i) is int for i in output):
        return 0.0
    i, j = output
    if i == j or not (0 <= i < len(a) and 0 <= j < len(a)):
        return 0.0
    return 1.0 if a[i] + a[j] == t else 0.0
''',
        reference=_two_sum_feasible,
        reference_doc="True if some pair sums to t, else False",
        small=({"n_min": 1, "n_max": 50},),
        large=({"n_min": 150_000, "n_max": 200_000},),
    ),
    Problem(
        name="n_queens",
        entry="n_queens",
        stub="def n_queens(n: int) -> list[int]:",
        description=(
            "Place n queens on an n x n board so that no two attack each other. Return a list "
            "`cols` where cols[r] is the column (0-based) of the queen in row r."
        ),
        generate_src='''def generate(rng, n_min, n_max):
    return [rng.randint(n_min, n_max)]
''',
        check_src='''def check(args, output, expected):
    n = args[0]
    if not isinstance(output, list) or len(output) != n or not all(type(c) is int for c in output):
        return 0.0
    if sorted(output) != list(range(n)):
        return 0.0
    if len({r + c for r, c in enumerate(output)}) != n or len({r - c for r, c in enumerate(output)}) != n:
        return 0.0
    return 1.0
''',
        reference=lambda n: None,
        reference_doc="None (unused; check validates the placement directly)",
        small=({"n_min": 4, "n_max": 8},),
        large=({"n_min": 2_000, "n_max": 5_000},),
    ),
    Problem(
        name="count_primes",
        entry="count_primes",
        stub="def count_primes(n: int) -> int:",
        description="Return the number of primes strictly less than n.",
        generate_src='''def generate(rng, n_min, n_max):
    return [rng.randint(n_min, n_max)]
''',
        check_src=EXACT_CHECK,
        reference=_count_primes,
        reference_doc="the correct answer",
        small=({"n_min": 2, "n_max": 1_000},),
        large=({"n_min": 3_000_000, "n_max": 5_000_000},),
    ),
    Problem(
        name="knapsack",
        entry="knapsack",
        stub="def knapsack(weights: list[int], values: list[int], capacity: int) -> int:",
        description=(
            "Each item i has weight weights[i] and value values[i] and can be taken at most once. "
            "Return the largest total value of items whose total weight is at most `capacity`."
        ),
        generate_src='''def generate(rng, n_min, n_max, w_max, capacity):
    n = rng.randint(n_min, n_max)
    weights = [rng.randint(1, w_max) for _ in range(n)]
    values = [rng.randint(1, 1000) for _ in range(n)]
    return [weights, values, capacity]
''',
        check_src=EXACT_CHECK,
        reference=_knapsack,
        reference_doc="the correct answer",
        small=({"n_min": 1, "n_max": 12, "w_max": 50, "capacity": 100},),
        large=({"n_min": 90, "n_max": 100, "w_max": 1_000, "capacity": 10_000},),
    ),
    Problem(
        name="coin_change",
        entry="coin_change",
        stub="def coin_change(coins: list[int], amount: int) -> int:",
        description=(
            "Return the fewest coins (each denomination usable any number of times) that sum to "
            "exactly `amount`, or -1 if it cannot be done."
        ),
        generate_src='''def generate(rng, k_min, k_max, c_max, amount_max):
    coins = sorted(set(rng.randint(2, c_max) for _ in range(rng.randint(k_min, k_max))))
    return [coins, rng.randint(1, amount_max)]
''',
        check_src=EXACT_CHECK,
        reference=_coin_change,
        reference_doc="the correct answer",
        small=({"k_min": 1, "k_max": 3, "c_max": 10, "amount_max": 30},),
        large=({"k_min": 5, "k_max": 8, "c_max": 500, "amount_max": 100_000},),
    ),
    Problem(
        name="two_coloring",
        entry="two_coloring",
        stub="def two_coloring(n: int, edges: list[list[int]]) -> list[int] | None:",
        description=(
            "Vertices are 0..n-1 and `edges` are undirected. Return a list `color` of 0s and 1s, "
            "one per vertex, such that every edge joins vertices of different colors, or None "
            "if that is impossible."
        ),
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    side = [rng.randint(0, 1) for _ in range(n)]
    edges = []
    for _ in range(n + n // 2):
        u, v = rng.randrange(n), rng.randrange(n)
        if side[u] != side[v]:
            edges.append([u, v])
    if rng.random() < 0.5 and n >= 3:
        a, b, c = rng.sample(range(n), 3)
        edges += [[a, b], [b, c], [c, a]]
    return [n, edges]
''',
        check_src='''def check(args, output, expected):
    n, edges = args
    if output is None:
        return 0.0 if expected else 1.0
    if not isinstance(output, list) or len(output) != n or any(c not in (0, 1) or type(c) is not int for c in output):
        return 0.0
    return 1.0 if all(output[u] != output[v] for u, v in edges) else 0.0
''',
        reference=_is_bipartite,
        reference_doc="True if a valid coloring exists, else False",
        small=({"n_min": 1, "n_max": 8},),
        large=({"n_min": 80_000, "n_max": 100_000},),
    ),
    Problem(
        name="tsp_tour",
        entry="tsp_tour",
        stub="def tsp_tour(points: list[list[int]]) -> list[int]:",
        description=(
            "Return an order in which to visit every point exactly once (a permutation of "
            "0..len(points)-1). The tour returns to its start; shorter tours score higher."
        ),
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    return [[[rng.randint(0, 10_000), rng.randint(0, 10_000)] for _ in range(n)]]
''',
        check_src='''def check(args, output, expected):
    points = args[0]
    n = len(points)
    if not isinstance(output, list) or sorted(output) != list(range(n)) or not all(type(i) is int for i in output):
        return 0.0
    length = sum(math.dist(points[output[i]], points[output[i - 1]]) for i in range(n))
    return 1.0 if length <= expected else expected / length
''',
        reference=_tsp_target,
        reference_doc=(
            "a target length: the optimal tour length if len(points) <= 8, else 0.93 times "
            "the length of the nearest-neighbour tour starting from point 0"
        ),
        small=({"n_min": 5, "n_max": 8},),
        large=({"n_min": 300, "n_max": 500},),
    ),
    Problem(
        name="longest_palindrome",
        entry="longest_palindrome",
        stub="def longest_palindrome(s: str) -> int:",
        description="Return the length of the longest palindromic substring of `s`.",
        generate_src='''def generate(rng, n_min, n_max):
    n = rng.randint(n_min, n_max)
    chars = []
    while len(chars) < n:
        chars += [rng.choice("ab")] * rng.randint(1, max(1, n // 4))
    return ["".join(chars[:n])]
''',
        check_src=EXACT_CHECK,
        reference=_longest_palindrome,
        reference_doc="the correct answer",
        small=({"n_min": 1, "n_max": 30},),
        large=({"n_min": 80_000, "n_max": 100_000},),
    ),
)

# Held-out problems for the test split; the rest are train.
TEST_PROBLEMS = frozenset({"count_inversions", "knapsack", "two_coloring", "tsp_tour"})
