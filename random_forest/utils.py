import numpy as np
import numba


@numba.njit
def fisher_yates_shuffle(arr, n, continue_at=0):
    """Perform a partial shuffle Fisher-Yates shuffle of the first
    `n` values in `arr`, drawing without replacement from values in
    the whole array.
    
    Arguments
    ---------
    arr : 1d array
        The values that we want to draw values from
    n : int
        The number of values to draw. These will be placed at the
        beginning of `arr`.

    Returns
    -------
    Nothing, the order of the values in `arr` is changed.
    """
    for i in range(continue_at, min(arr.size, n)):
        idx = np.random.randint(arr.size - i) + i
        arr[i], arr[idx] = arr[idx], arr[i]


@numba.njit
def enlarge_first_dimension(x, size=-1):
    """Enlarge the first dimension of x
    
    Arguments
    ---------
    x : ndarray
    size : [-1] | positive int
        The new size of the first dimension of x. If this is negative then
        the first dimension size will be doubled (i.e. size = x.shape[0] * 2).
    
    Returns
    -------
    ret : array with first dimension shape given by size. All of the elements
    of `x` are present in `ret` in their original position, with garbage
    values in the new positions created.
    """
    if size < 0:
        size = x.shape[0] * 2
    elif size < x.shape[0]:
        raise ValueError("`size` must be >= than dim-0 size of `x`.")
    ret = np.empty((size,) + x.shape[1:], dtype=x.dtype)
    ret[:x.shape[0]] = x
    return ret


@numba.njit
def mean_and_sse(arr):
    """Compute the mean and variance of a 1d array in linear time.

    Returns
    -------
    m, v : float
        The values Mean(arr[sorter]), Var(arr[sorter])

    Notes
    -----
    These values are computed in linear time.
    """
    m = arr[0]  # The running mean of the array
    s = 0.0
    for k in range(1, arr.size):
        val = arr[k]
        m_new = m + ((val - m) / (k + 1))
        s = s + ((val - m) * (val - m_new))
        m = m_new
    
    return m, s