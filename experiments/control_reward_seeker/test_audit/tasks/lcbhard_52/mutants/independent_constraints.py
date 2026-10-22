# treats the two limits independently: min(#cheapest-by-sweetness fitting x, #cheapest-by-saltiness fitting y) + 1
def max_dishes_eaten(n, x, y, dishes):
    def cnt(vals, lim):
        s = c = 0
        for v in sorted(vals):
            if s + v > lim:
                break
            s += v; c += 1
        return c
    k = min(cnt([d[0] for d in dishes], x), cnt([d[1] for d in dishes], y))
    return min(k + 1, n)
