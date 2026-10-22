# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Find the max k such that some k dishes have total sweetness <= x and saltiness <= y;
# eating those first, then any other dish, gives min(k + 1, n).
# DP over count j keeping the Pareto frontier of (sweetness, saltiness) pairs.
from typing import List, Tuple


def _pareto(pts):
    pts.sort()
    out = []
    best = None
    for a, b in pts:
        if best is None or b < best:
            out.append((a, b))
            best = b
    return out


def max_dishes_eaten(n: int, x: int, y: int, dishes: List[Tuple[int, int]]) -> int:
    fronts = [[(0, 0)]]
    for s, t in dishes:
        if s > x or t > y:
            continue
        new_top = None
        for j in range(len(fronts) - 1, -1, -1):
            cand = [(a + s, b + t) for a, b in fronts[j] if a + s <= x and b + t <= y]
            if not cand:
                continue
            if j + 1 < len(fronts):
                fronts[j + 1] = _pareto(fronts[j + 1] + cand)
            else:
                new_top = _pareto(cand)
        if new_top:
            fronts.append(new_top)
    k = len(fronts) - 1
    return min(k + 1, n)
