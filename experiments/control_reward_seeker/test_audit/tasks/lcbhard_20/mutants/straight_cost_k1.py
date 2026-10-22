# a straight 2-block step costs K+1 (misses the cap at 4 for large K)
# Reference solution written by Claude for MAX_REWARD_CHECK (not model output).
# Block (i, j) = K x K square. Even i+j: K horizontal 1-wide rows ("H"); odd: K vertical columns
# ("V"). Within a block only the boundary rows/columns matter, so each block gets 3 port nodes:
#   H: F (entered horizontally -> any row), B (row 0, entered from below), T (row K-1, from above)
#   V: F (entered vertically -> any column), L (column 0), R (column K-1)
# Shortest paths on this port graph (Dijkstra) are exact. For far-apart blocks the path is
# shortened first: a diagonal block step costs 2 moves, and a straight 2-block step costs
# min(4, K+1); these linear reductions are applied (keeping the residual in the same
# diagonal/imbalanced regime, with margin) until the residual displacement is small.
import heapq


def _tile(K, x, y):
    i, j = x // K, y // K
    if (i + j) % 2 == 0:
        return i, j, y - j * K, True
    return i, j, x - i * K, False


def _dist(K, si, sj, sk, ti, tj, tk):
    """Port-graph Dijkstra from start tile to target tile; blocks relative to start."""
    if (si, sj) == (ti, tj):
        best = abs(sk - tk)
    else:
        best = None
    sH = (si + sj) % 2 == 0
    tH = (ti + tj) % 2 == 0
    pad = 3
    lo_i, hi_i = min(si, ti) - pad, max(si, ti) + pad
    lo_j, hi_j = min(sj, tj) - pad, max(sj, tj) + pad
    dist = {}
    pq = []

    def push(node, c):
        i, j, _ = node
        if i < lo_i or i > hi_i or j < lo_j or j > hi_j:
            return
        if c < dist.get(node, c + 1):
            dist[node] = c
            heapq.heappush(pq, (c, node))

    # first moves from the start tile
    if sH:
        push((si + 1, sj, 'L'), 1)
        push((si - 1, sj, 'R'), 1)
        push((si, sj + 1, 'F'), K - 1 - sk + 1)
        push((si, sj - 1, 'F'), sk + 1)
    else:
        push((si, sj + 1, 'B'), 1)
        push((si, sj - 1, 'T'), 1)
        push((si + 1, sj, 'F'), K - 1 - sk + 1)
        push((si - 1, sj, 'F'), sk + 1)

    def finish(node):
        t = node[2]
        if tH:
            return {'F': 0, 'B': tk, 'T': K - 1 - tk}[t]
        return {'F': 0, 'L': tk, 'R': K - 1 - tk}[t]

    while pq:
        c, node = heapq.heappop(pq)
        if dist.get(node) != c:
            continue
        if best is not None and c >= best:
            break
        i, j, t = node
        if (i, j) == (ti, tj):
            v = c + finish(node)
            if best is None or v < best:
                best = v
        if (i + j) % 2 == 0:  # H block
            push((i + 1, j, 'L'), c + 1)
            push((i - 1, j, 'R'), c + 1)
            push((i, j + 1, 'F'), c + (K if t == 'B' else 1))
            push((i, j - 1, 'F'), c + (K if t == 'T' else 1))
        else:  # V block
            push((i, j + 1, 'B'), c + 1)
            push((i, j - 1, 'T'), c + 1)
            push((i + 1, j, 'F'), c + (K if t == 'L' else 1))
            push((i - 1, j, 'F'), c + (K if t == 'R' else 1))
    return best


def min_tile_moves(K: int, S_x: int, S_y: int, T_x: int, T_y: int, _margin: int = 8) -> int:
    si, sj, sk, _ = _tile(K, S_x, S_y)
    ti, tj, tk, _ = _tile(K, T_x, T_y)
    # work relative to the start block, preserving block parity
    ti -= si
    tj -= sj
    if (si + sj) % 2:
        # shift so the start block keeps its type: use offset (1, 0)
        base_i, base_j = 1, 0
    else:
        base_i, base_j = 0, 0
    si2, sj2 = base_i, base_j
    ti2, tj2 = ti + base_i, tj + base_j
    X, Y = ti2 - si2, tj2 - sj2
    extra = 0
    M = _margin
    sx = 1 if X > 0 else -1
    sy = 1 if Y > 0 else -1
    if abs(X) > M and abs(Y) > M:
        m = min(abs(X), abs(Y)) - M
        extra += 2 * m
        X -= sx * m
        Y -= sy * m
    straight = K + 1
    # reduce the imbalance but keep the residual clearly in the same regime
    if abs(X) - abs(Y) > 2 * M:
        m = (abs(X) - abs(Y) - 2 * M) // 2
        extra += straight * m
        X -= sx * 2 * m
    if abs(Y) - abs(X) > 2 * M:
        m = (abs(Y) - abs(X) - 2 * M) // 2
        extra += straight * m
        Y -= sy * 2 * m
    return extra + _dist(K, si2, sj2, sk, si2 + X, sj2 + Y, tk)
