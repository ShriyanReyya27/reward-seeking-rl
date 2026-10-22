# closed form: 2 moves per diagonal block step, 2 per straight block step, plus in-block offsets ignored
def min_tile_moves(K, S_x, S_y, T_x, T_y):
    def tile(x, y):
        i, j = x // K, y // K
        return i, j, (y - j * K) if (i + j) % 2 == 0 else (x - i * K)
    si, sj, sk = tile(S_x, S_y)
    ti, tj, tk = tile(T_x, T_y)
    if (si, sj) == (ti, tj):
        return min(abs(sk - tk), 2)
    dx, dy = abs(si - ti), abs(sj - tj)
    return 2 * max(dx, dy)
