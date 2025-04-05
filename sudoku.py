import time
import numpy as np
from itertools import product

def giai_sudoku(size, grid):
    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])

    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)  # vùng chứa số
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = rang_buoc(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                chon(X, Y, (i, j, n))
    for dap_an in giai(X, Y, []):
        for (r, c, n) in dap_an:
            grid[r][c] = n
        yield grid


def rang_buoc(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y


def giai(X, Y, dap_an):
    if not X:
        yield list(dap_an)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            dap_an.append(r)
            cols = chon(X, Y, r)
            for s in giai(X, Y, dap_an):
                yield s
            bo_chon(X, Y, r, cols)
            dap_an.pop()


def chon(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def bo_chon(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def dieu_kien_giai(mang_o_so):
    if mang_o_so.count('0') >= 80:
        return None, None

    start = time.time()

    # chuyển chuỗi thành mảng 9x9
    arr = []
    for i in mang_o_so:
        arr.append(int(i))

    arr = np.array(arr, dtype=int)
    arr = np.reshape(arr, (9, 9))
    try:
        ans = list(giai_sudoku(size=(3, 3), grid=arr))[0]
        s = ""
        for a in ans:
            s += "".join(str(x) for x in a)
        return s, "Giai trong %.4fs" % (time.time() - start)
    except:
        return None, None
