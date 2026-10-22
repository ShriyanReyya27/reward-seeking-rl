# ignores diagonal split: each unit cell counted fully black/white by colour of its centre
def calculate_black_area_doubled(A, B, C, D):
    # doubled area of black full cells in [0,x)x[0,y) via periodicity (period 4 in x, 2 in y)
    import math
    def black(i, j):
        x, y = i + 0.5, j + 0.5
        return (math.floor(x) + math.floor(y / 2) + math.floor((x + y) / 2)) % 2 == 0
    pat = [[2 if black(i, j) else 0 for j in range(2)] for i in range(4)]
    def F(x, y):
        tot = 0
        for r in range(4):
            cx = (x - r + 3) // 4 if x > r else 0
            for s in range(2):
                cy = (y - s + 1) // 2 if y > s else 0
                tot += cx * cy * pat[r][s]
        return tot
    sx, sy = 4 * 10 ** 9, 2 * 10 ** 9
    A += sx; C += sx; B += sy; D += sy
    return F(C, D) - F(A, D) - F(C, B) + F(A, B)
