from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist


def dtw(x, y, dist=lambda a,b: abs(b-a), w=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int w: percentage of warping allowed each side of the diagonal, e.g. 0 = no warping allowed, 1 = unlimited warping allowed
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    len_x, len_y = len(x), len(y)

    assert len_x, "Length of array x cannot be 0"
    assert len_y, "Length of array y cannot be 0"

    # Calculate r - the maximum distance from the diagonal
    r = int(w*len_x)
    print("r = %d" % r)

    # Calculate (partial) distance matrix
    D = full((len_x, len_y), inf)
    for i in range(len_x):
        D[i, i] = dist(x[i], y[i])
        for delt in range(r + 1):
            min_delt = min(i + delt, len_x - 1)
            D[i + min_delt, i] = dist(x[min_delt], y[i])
            D[i, i + min_delt] = dist(x[i], y[min_delt])

    print(D)

    # D0 = zeros((len_x + 1, len_y + 1))
    # D0[0, 1:] = inf
    # D0[1:, 0] = inf
    # D1 = D0[1:, 1:]  # D1 is a reference to a subset of D0, not a copy
    # for i in range(len_x):
    #     for j in range(len_y):
    #         D1[i, j] = dist(x[i], y[j])
    # cost_matrix = D1.copy()
    #
    #
    # # Note: Change from D0 and D1 in here, where D0[i][j] + warp represents
    # #       the top-left, top and left cells (where warp=1)
    # for i in range(len_x):
    #     for j in range(len_y):
    #         min_list = [D0[i, j]]
    #         for k in range(1, warp + 1): # k is confusing, makes it look like a 3rd dimension
    #             i_k = min(i + k, len_x - 1)
    #             j_k = min(j + k, len_y - 1)
    #             min_list += [D0[i_k, j], D0[i, j_k]]
    #         D1[i, j] += min(min_list)
    # if len(x)==1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    #     path = _traceback(D0)
    # return D1[-1, -1] / sum(D1.shape), cost_matrix, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r - 1), j],
                             D0[i, min(j + k, c - 1)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
