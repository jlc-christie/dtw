from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist


def dtw(x, y, dist=lambda a,b: abs(b-a), w=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    Complexity:
      time  - O(nw)
      space - O(n^2)

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int w: percentage of warping allowed each side of the diagonal, e.g. 0 = no warping allowed, 0.10 = 10% warping allowed
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the warp path.
    """
    len_x, len_y = len(x), len(y)

    assert len_x, "Length of array x cannot be 0"
    assert len_y, "Length of array y cannot be 0"

    # Calculate r - the maximum distance from the diagonal
    r = int(w*len_x)
    print("r = %d" % r)

    # Calculate cost matrix (within bounds, if applicable)
    cost_matrix = zeros((len_x,len_y))
    for i in range(len_x):
        cost_matrix[i,i] = dist(x[i], y[i])
        _r = r # temp variable for r if exceeds limits of matrix
        if i + r >= len_x:
            _r = len_x - i - 1
        for delt in range(1, _r + 1):
            cost_matrix[i + delt,i] = dist(x[i + delt], y[i])
            cost_matrix[i,i + delt] = dist(x[i], y[i + delt])

    # Matrix to calculate path, at the end it will be the accumulated cost matrix
    D = full((len_x+1, len_y+1), inf)
    D[0,0] = 0
    for i in range(1,len_x+1):
        for j in range(max(1,i-r), min(len_x+1,i+r+1)):
            cost = cost_matrix[i-1,j-1]
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])

    dist = D[-1,-1]
    print("Distance = %f" % (D[-1,-1]))

    # Is this path correct? If so, is plot correct?
    path = _traceback(D)

    return dist, cost_matrix, D, path

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
