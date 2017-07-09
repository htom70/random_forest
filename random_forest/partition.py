import numpy as np
import numba

from .utils import fisher_yates_shuffle
from .sort import introsort


@numba.njit
def sse(x):
    m = x.mean()
    total = numba.float32(0.0)
    for val in x:
        total += (val - m) * (val - m)
    return total


from collections import OrderedDict
spec = OrderedDict()
spec["index"] = numba.int32
spec["left_mean"] = numba.float32
spec["left_sse"] = numba.float32
spec["right_mean"] = numba.float32
spec["right_sse"] = numba.float32
spec["dim"] = numba.int32
spec["split_value"] = numba.float32
spec["left_is_sorted"] = numba.boolean

@numba.jitclass(spec)
class PartitionResult:
    def __init__(self):
        pass


local_types_init_partition = { 'n': numba.int32,
  "m_left_new": numba.float32, "m_right_new": numba.float32 }

@numba.njit(locals=local_types_init_partition)
def initialize_partition(y, pr, mean_total, sse_total):
    """Initialize a PartitionResult object to the partition: y[:1], y[1:]
    
    Arguments
    ---------
    y : 1d array
        Data vector
    pr : PartitionResult object
    mean_total, sse_total: float32
        The mean and sum-of-squared errors in y.
    
    Returns
    -------
    Nothing, pr is mutated to hold the result.
    """
    n = y.size
    pr.index = numba.int32(1)
    pr.left_mean = y[0]
    pr.right_mean = (n * mean_total - y[0]) / (n - 1)
    pr.left_sse = numba.float32(0.0)
    pr.right_sse = sse_total - ((y[0] - pr.right_mean) * (y[0] - mean_total))


local_types_best_partition = {
    'best_sse': numba.float32, 'n': numba.int32,
    'k': numba.int32, 'val': numba.float32,
    'l_mean': numba.float32, 'l_mean_new': numba.float32,
    'r_mean': numba.float32, 'r_mean_new': numba.float32,
    'l_sse': numba.float32, 'r_sse': numba.float32}

@numba.njit(locals=local_types_best_partition)
def _best_partition(
  x, pr, min_samples_leaf, mean_total=np.nan, sse_total=np.nan):
    if np.isnan(mean_total):
        mean_total = np.mean(x)
    
    if np.isnan(sse_total):
        sse_total = sse(x)
    
    initialize_partition(x, pr, mean_total, sse_total)
    n = x.size
    l_mean = pr.left_mean
    l_mean_new = 0.0
    r_mean = pr.right_mean
    r_mean_new = 0.0
    l_sse = pr.left_sse
    r_sse = pr.right_sse
    
    best_err = np.inf # l_sse + r_sse
    
    for i in range(1, n - min_samples_leaf):
        val = x[i]
        
        l_mean_new = l_mean + ((val - l_mean) / (i + 1))
        l_sse = l_sse + (val - l_mean) * (val - l_mean_new)
        l_mean = l_mean_new
        
        r_mean_new = ((n - i) * r_mean - val) / (n - i - 1)
        r_sse = r_sse - (val - r_mean_new) * (val - r_mean)
        r_mean = r_mean_new
        
        if i + 1 >= min_samples_leaf and l_sse + r_sse < best_err:
            best_err = l_sse + r_sse
            pr.left_mean = l_mean
            pr.right_mean = r_mean
            pr.left_sse = l_sse
            pr.right_sse = r_sse
            pr.index = i + 1


@numba.njit
def check_partition(X, threshold, dim, index):
    for _x in X[:index, dim]:
        assert _x <= threshold
    
    for _x in X[index:, dim]:
        assert _x >= threshold


@numba.njit
def partition_on_feature(X, y, threshold, dim, index):
    """Reorder rows of X and elements of y so that:
        X[:partition_index, dim] <= threshold,  and
        X[partition_index:, dim] > threshold

    Similar to a Hoare quicksort partition, except the pivot is known.
    """
    n, n_features = X.shape
    i = -1
    tmp = X[0, 0]
    
    while index > 1 and X[index - 1, dim] == X[index, dim]:
        index -= 1
    
    j = index - 1
    
    while True:
        
        i += 1
        while i < index and X[i, dim] < threshold:
            i += 1
        
        j += 1
        while j < n and X[j, dim] >= threshold:
            j += 1
        
        if i >= index or j >= n:
            break
        
        y[i], y[j] = y[j], y[i]
        for k in range(n_features):
            X[i, k], X[j, k] = X[j, k], X[i, k]
            #tmp = X[i, k]
            #X[i, k] = X[j, k]
            #X[j, k] = tmp

    # for i in range(index):
    #     _x = X[i, dim]
    #     if _x > threshold:
    #         print(_x, threshold, X[i - 1, dim], X[i + 1, dim])
    #         raise ValueError("bad partition. large value in bottom")
    #
    # for i in range(index, n):
    #     _x = X[i, dim]
    #     if _x < threshold:
    #         print(_x, threshold, X[i - 1, dim], X[i + 1, dim])
    #         raise ValueError("bad partition, small value in top.")


@numba.njit
def is_sorted(x):
    if x.size <= 1:
        return True
    for i in range(x.size - 1):
        if x[i] > x[i + 1]:
            return False
    return True


@numba.njit
def assert_is_sorted(x):
    if x.size <= 1:
        return
    
    for i in range(x.size - 1):
        if x[i] > x[i + 1]:
            print("AssertionError: ", x[i], '>', x[i + 1])
            raise AssertionError("Array is not sorted")


@numba.njit(locals={'i': numba.uint32})
def arange(n):
    r = np.empty(n, dtype=np.uint32)
    for i in range(n):
        r[i] = i
    return r


@numba.njit
def partition_on_argsort_front(X, y, argsort, n):
    """Put the first n elements of X and y in sorted order."""
    j = argsort[0]
    n_features = X.shape[1]
    tmp_x = X[0, 0]
    tmp_y = y[0]
    
    for i in range(n):
        j = argsort[i]
        while j < i:
            j = argsort[j]

        for f in range(n_features):
            X[i, f], X[j, f] = X[j, f], X[i, f]
            #tmp_x = X[i, f]
            #X[i, f] = X[j, f]
            #X[j, f] = tmp_x
        y[i], y[j] = y[j], y[i]
        #tmp_y = y[i]
        #y[i] = y[j]
        #y[j] = tmp_y


@numba.njit
def partition_on_argsort_back(X, y, argsort, n):
    """Put the last n elements of X and y in sorted order."""
    j = argsort[0]
    n_features = X.shape[1]
    tmp_x = X[0, 0]
    tmp_y = y[0]
    
    for i in range(y.size - 1, n - 1, -1):
        j = argsort[i]
        while j > i:
            j = argsort[j]
        
        for f in range(n_features):
            X[i, f], X[j, f] = X[j, f], X[i, f]
            #tmp_x = X[i, f]
            #X[i, f] = X[j, f]
            #X[j, f] = tmp_x
        y[i], y[j] = y[j], y[i]
        #tmp_y = y[i]
        #y[i] = y[j]
        #y[j] = tmp_y


@numba.njit
def best_partition(X, y, max_features, min_samples_leaf, dims_permutation, how,
    mean_of_y, sse_of_y, best, temp, sorted_dim=-1):
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
    best_err = np.inf
    fisher_yates_shuffle(dims_permutation, max_features)
    
    best_i_sort = np.zeros(0, dtype=np.uint32)
    best_y_sort = np.zeros(0, dtype=y.dtype)
    
    for i_dim in range(max_features):
        
        dim = dims_permutation[i_dim]
        
        if sorted_dim == dim:
            i_sort = np.zeros(0, dtype=np.uint32)
            x_sort = X[:, dim]
            y_sort = y
        else:
            # i_sort = np.argsort(X[:, dim])
            # x_sort = X[i_sort, dim]
            x_sort = X[:, dim].copy()
            i_sort = arange(x_sort.size)
            introsort(x_sort, i_sort)
            y_sort = y[i_sort]
        
        _best_partition(y_sort, temp, min_samples_leaf, mean_of_y, sse_of_y)
        
        if temp.left_sse + temp.right_sse < best_err:
            best, temp = temp, best
            best_err = best.left_sse + best.right_sse
            best.dim = dim
            best.split_value = x_sort[best.index]
            best_i_sort = i_sort
            best_y_sort = y_sort
    
    #partition_on_feature(X, y, best.split_value, best.dim, best.index)
    
    if sorted_dim != best.dim:
        # if best.index > y.size // 2:
        #     partition_on_argsort_back(X, y, i_sort, best.index)
        #     best.left_sorted = False
        # else:
        #     partition_on_argsort_front(X, y, i_sort, best.index)
        #     best.left_sorted = True
        X[:] = X[best_i_sort]
        #y[:] = y[best_i_sort]
        y[:] = best_y_sort
    
    return best