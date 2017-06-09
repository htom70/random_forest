import numpy as np
import numba

from .utils import fisher_yates_shuffle
from .sort import timsort_


@numba.njit
def find_minimum_sse_partition(y, MIN_SAMPLES_LEAF, mean_of_y, sse_of_y,
                               partition_result):
    """Find the index of an array that: SSE(y[:i]) + SSE(y[i:]) is minimized

    Arguments
    ---------
    y : 1d array
    sorter : 1d array of ints
        Gives the order of elements of `y` that would result in a sorted version
        of `y`.
    mean_of_y : float
        Precomputed mean of `y`
    sse_of_y : float
        Precomputed sum of squared errors of `y`
    partition_result : PartitionResult object
        The results of the partitioning will be put in here.
        The following fields will be populated:
            best_index, left_mean, left_sse, right_mean, right_sse

    Returns
    -------
    best_index : int
        The index that minimized the residual sum of squares by
        splitting y into y[best_index:], y[best_index:].

    Raises
    ------
    ValueError if y.size < 2 or y.size / 2 < MIN_LEAF_SIZE
    ValueError if y.size is less than MIN_SAMPLES_LEAF
    """
    n = len(y)
    if n < 2:
        raise ValueError("Length of array must be >= 2")
    elif MIN_SAMPLES_LEAF > n / 2:
        print('n', n, ',  MIN_SAMPLES_LEAF', MIN_SAMPLES_LEAF)
        raise ValueError("Length of array must be >= (2 * MIN_LEAF_SIZE).")
    elif n == 2:
        partition_result[0] = y[0]
        partition_result[1] = 0.0
        partition_result[2] = y[1]
        partition_result[3] = 0.0
        return 1
    
    m_left = np.float32(y[0])
    m_left_new = np.float32(0.0)
    m_right_new =  np.float32( ((n * mean_of_y) - m_left) / (n - 1)  )
    s_right = np.float32(sse_of_y - ((m_left-m_right_new)*(m_left-mean_of_y)))
    m_right = np.float32( m_right_new )
    s_left = np.float32( 0.0 )
    best_sse = np.float32( 0.0 )
    best_index = np.int32(0)
    
    if MIN_SAMPLES_LEAF == 1:
        best_sse = s_left + s_right
        best_index = 1
        partition_result[0] = m_left
        partition_result[1] = s_left
        partition_result[2] = m_right
        partition_result[3] = s_right
    else:
        best_sse = np.inf

    # i is one less than the split value
    for i in range(1, n - MIN_SAMPLES_LEAF):
        k = i + 1  # Number of elements in left part
        j = n - i # Current size (including y) of right part
        
        y_i = y[i]
        
        m_left_new = m_left + ((y_i - m_left) / k)
        m_right_new = ((j * m_right) - y_i) / (j - 1)
        
        s_left = s_left + ((y_i - m_left) * (y_i - m_left_new))
        s_right = s_right - ((y_i - m_right_new) * (y_i - m_right))
        
        m_left = m_left_new
        m_right = m_right_new
        
        if k >= MIN_SAMPLES_LEAF and s_left + s_right < best_sse:
            best_sse = s_left + s_right
            best_index = k
            partition_result[0] = m_left
            partition_result[1] = s_left
            partition_result[2] = m_right
            partition_result[3] = s_right
    return best_index


@numba.njit
def partition_on_feature(X, y, threshold, dim, partition_index):
    """Reorder rows of X and elements of y to that:
        X[:partition_index, dim] <= threshold,  and
        X[partition_index:, dim] > threshold
    """
    n, n_features = X.shape
    j = partition_index
    tmp = np.empty(n_features, dtype=X.dtype)
    for i in range(partition_index):
        if X[i, dim] > threshold:
            # Exchange with the first low value in the high partition
            while j < n - 1 and X[j, dim] > threshold:
                j += 1
            y[i], y[j] = y[j], y[i]
            tmp = X[i].copy()
            X[i, :] = X[j, :]
            X[j, :] = tmp


@numba.njit
def best_partition(X, y, max_features, min_samples_leaf, dims_permutation, how,
    mean_of_y, sse_of_y, best_partition_result, temp_partition_result,
    last_best_dim=-1):
    """Find the best split along multiple (random) features.

    Features are selected until at least one is valid, possibly
    exceeding the `params.max_features`.

    Arguments
    ---------
    X : n-by-m array
        Each column of X corresponds to a different feature.
    y : length-n array
        The target variables to model.
    max_features : int
        The number of features to consider during each split.
    min_samples_leaf : int
        The minimum number of samples required to build a leaf node
    dims_permutation : array of integers
        Should containing all the values in the half-open range [0:m).
    how : uint8 array with length m
        Flag indicating how to partition the data for each dimension.
        0 - SSE.
        As of now, no other impurities are supported (Gini, etc...)
    
    Returns
    -------
    best_dim - int
        A dimension >= 0 if a valid split was found. Otherwise -1.
    split_index - int
        The index at which to split: X[:split_index], X[split_index:]

    Notes
    -----
    If a valid partition is found than this function mutates X and y.
    The rows of X and corresponding values of y are reordered so that
    X[dim, :] is sorted.
    """
    n, n_features = X.shape
    best_err = np.inf
    fisher_yates_shuffle(dims_permutation, max_features, continue_at=0)
    for i_dim in range(max_features):
        dim = dims_permutation[i_dim]
        
        if dim == last_best_dim:
            # This dimension is already sorted along X[:, dim]
            x_sort = X[:, dim]
            y_sort = y
        else:
            x_sort = X[:, dim].copy()
            y_sort = y.copy()
            timsort_(x_sort, y_sort)
        
        index = find_minimum_sse_partition(y_sort, min_samples_leaf,
                                         mean_of_y, sse_of_y,
                                         temp_partition_result)
        err = temp_partition_result[1] + temp_partition_result[3]
        
        if err < best_err:
            best_threshold = x_sort[index - 1]
            best_err = err
            best_index = index
            best_dim = dim
            best_partition_result[:] = temp_partition_result
            found_valid_split = True
    
    partition_on_feature(X, y, best_threshold, best_dim, best_index)
    
    return best_dim, best_index