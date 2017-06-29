"""
Modified implementation of indirect timsort.
"""
#TODO: Technically not yet (need to implement galloping mode).

import numba
import numpy as np
from collections import OrderedDict


@numba.njit
def enlarge_array(arr, size=-1):
    """Enlarge an array to given size, or double the size
    if no size is given.
    """
    if size < 0:
        size = arr.size * 2
    elif size < arr.size:
        raise ValueError("final size less than current size.")
    ret = np.empty(size, dtype=arr.dtype)
    ret[:arr.size] = arr
    return ret


spec = OrderedDict()
spec['_n'] = numba.uint32  # Current size of the array
spec['_indexes'] = numba.uint32[:]

@numba.jitclass(spec)
class Stack:
    """FIFO data structure that stores uint32s. Doesn't reallocate."""
    
    def __init__(self, capacity):
        self._indexes = np.empty(max(capacity, 2), np.uint32)
        self._n = 0
    
    def push(self, val):
        if self._n >= self._indexes.size:
            self._indexes = enlarge_array(self._indexes)
        self._indexes[self._n] = val
        self._n += 1
        return self
    
    def pop(self):
        self._n -= 1
        return self._indexes[self._n]
    
    def remove(self, position):
        idx = self._indexes
        for i in range(position, self.n - 1):
            idx[i] = idx[i + 1]
        self._n -= 1
    
    @property
    def n(self):
        return self._n
    
    @property
    def arr(self):
        return self._indexes[:self._n]


def print_stack(stack):
    print(stack._indexes[:stack._n])


@numba.njit
def find_minrun(n):
    """ Find the minimum size that we should
    Taken directly from:
    <http://svn.python.org/projects/python/trunk/Objects/listsort.txt>
    """
    r = 0
    if not n >= 0:
        raise ValueError("array_size must be >= 0")
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


@numba.njit
def assert_sorted(arr):
    if arr.size <= 1:
        return
    for i in range(1, arr.size):
        if arr[i - 1] > arr[i]:
            raise AssertionError("array is not sorted.")


@numba.njit
def insertion_sort_indirect(labels, isort, key=0):
    """The key is the index of the first unsorted run.
    
    Every element of labels is sorted ascending below the key.
    """
    if key < 1:
        key += 1
    #assert_sorted(labels[0:key])
    if labels.shape[0] <= 1:
        return
    while key < labels.size:
        j = key
        while j >= 1 and labels[j] < labels[j - 1]:
            labels[j], labels[j - 1] = labels[j - 1], labels[j]
            isort[j], isort[j - 1] = isort[j - 1], isort[j]
            j -= 1
        key += 1


@numba.njit
def merge_indirect(labels, sort_order, i, j, k):
    """Merge the sorted subarrays labels[i:j] and labels[j:k].

    labels : uint32 array
        The integer values to sort on.
    isort : uint32 arra
        The original positions of each label. Is manipulated identically
        to labels so that the final isort array is an indirect sorter.
    i, j, k : int
        These are the partitions of the labels array that need to be sorted.
        We get away with passing in three indices instead of four
        (i.e. start0, end0, start1, end1) because the runs are always
        adjacent and end0 will always equal start1
    """
    # TODO: Add galloping (AKA exponential search to initial merges site).
    #assert_sorted(labels[i:j])
    #assert_sorted(labels[j:k])
    orig_i = i
    sz_a, sz_b = j - i, k - j
    a, a_isort = labels[i:j].copy(), sort_order[i:j].copy()
    b, b_isort = labels[j:k], sort_order[j:k]

    ia, ib = 0, 0
    
    write_count = 0
    while ia < sz_a and ib < sz_b:
        if a[ia] <= b[ib]:
            labels[i] = a[ia]
            sort_order[i] = a_isort[ia]
            i += 1
            ia += 1
            write_count += 1
        else:  # a[ia] > b[ib]:
            labels[i] = b[ib]
            sort_order[i] = b_isort[ib]
            i += 1
            ib += 1
            write_count += 1
    
    if ia != sz_a:
        labels[i:i + sz_a - ia] = a[ia:sz_a]
        sort_order[i:i + sz_a - ia] = a_isort[ia:sz_a]
        #assert write_count + sz_a - ia == k - orig_i
    elif ib != sz_b:
        labels[i:i + sz_b - ib] = b[ib:sz_b]
        sort_order[i:i + sz_b - ib] = b_isort[ib:sz_b]
        #assert write_count + sz_b - ib == k - orig_i
    else:
        raise RuntimeError("This should never be reached")


@numba.njit
def sort_run(arr, sort_order, start_idx, min_run):
    """Starting at `index` of `arr`, make at least `min_run` elements sorted.

    Arguments
    ---------
    arr : 1d array
    idx : location of `arr` to start sorting at
    min_run : The minimum number of values to sort

    Returns
    -------
    The index up to which `arr` has been sorted (starting at `index` of
    course). The maximum value that will be returned is array.shape[0]
    """
    idx = start_idx
    n = arr.size
    n_1 = n - 1
    if idx >= n_1:
        return n
    
    # Put idx after the already sorted elements
    while idx < n_1 and arr[idx] <= arr[idx + 1]:
        idx += 1
    
    # If min_run elements are already sorted...
    if idx >= start_idx + min_run or idx == arr.size:
        #assert not idx > arr.size
        return idx
    
    # Otherwise: insertion_sort up to min_run elements after idx
    else:
        fin = min(start_idx + min_run, n)
        insertion_sort_indirect(
            arr[start_idx:fin], sort_order[start_idx:fin], key=idx - start_idx)
        #assert_sorted(arr[start_idx:fin])
        return fin


@numba.njit
def collapse_merge(stack, labels, sort_order):
    """Ensure that the timsort invariants are met on the final 3 runs.

    stack : IndexStack instance
    labels : uint32 array
    sort_order : uint32 array

    Returns True if a merge was performed.
    Returns False if no merging is done.
    """
    n = stack.n
    if n <= 2:
        return False
    b, c, d, = stack.arr[n - 3:]  # Last three merge positions
    
    # Invariant condition number 2
    if c - b <= d - c:  # Distance between
        merge_indirect(labels, sort_order, b, c, d)
        stack.remove(n - 2)  # Remove the `c` position from the stack
        return True
    
    # Invariant condition number 1
    elif n > 3:
        a = stack.arr[n - 4]
        if b - a < d - b:  # Invariant number 1 is not met
            if b - a < d - c:  # Merge B with A when A is smaller than B
                merge_indirect(labels, sort_order, a, b, c)
                stack.remove(n - 3)  # Removing `b`
            else:  # Merge B with C
                merge_indirect(labels, sort_order, b, c, d)
                stack.remove(n - 2)  # Removing `c`
            return True
    return False  # All of the invariants are met


@numba.njit
def arange(n):
    ret = np.empty(n, dtype=np.int32)
    for i in range(n):
        ret[i] = i
    return ret


@numba.njit(nogil=True)
def timsort_(x, y):
    """Modified version of timsort that modifies (x, y) pairs so
    that x is sorted ascending.
    
    An indirect sort can be imitated by passing in np.arange(x.size)
    as the y argument.
    
    Arguments
    ---------
    x : 1d array (dtype doesn't matter)
        Labels that we want to sort
    y : 1d array (dtype doesn't matter)
        Previous indirect sorting on this array. This argument is for
        internal use only.
    
    Returns
    -------
    None, both x and y are mutated.
    
    Note
    ----
    `labels` will be sorted in place. Pass in a copy of `labels` if you do
    not want it to be mutated.
    """
    n = x.size
    #if sort_order is None:
    #    sort_order = arange(n)
    assert x.shape[0] == y.shape[0]
    
    min_run = find_minrun(n)
    stack = Stack(20)
    stack.push(0)
    index = 0
    while index != n:
        index = sort_run(x, y, index, min_run)
        stack.push(index)
        while stack.n >= 3 and collapse_merge(stack, x, y):
            pass
    
    while stack.n > 2:
        i, j, k = stack.arr[stack.n - 3:]
        merge_indirect(x, y, i, j, k)
        stack.remove(stack.n - 2)
    
    assert stack.arr[0] == 0 and stack.arr[1] == n