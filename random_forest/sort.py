# Sorting routines that simultaneously sort two arrays based on the values in
# the first array. In all cases these are implemented non-recursively
import numpy as np
import numba


@numba.njit
def insertion_sort(x, y, start, end):
    """start and end are inclusive"""
    for i in range(start + 1, end + 1):
        tmp_x = x[i]
        tmp_y = y[i]
        j = i - 1
        while j >= start and x[j] > tmp_x:
            x[j + 1] = x[j]
            y[j + 1] = y[j]
            j -= 1
        
        x[j + 1] = tmp_x
        y[j + 1] = tmp_y


@numba.njit
def median_of_3(arr):
    a, b, c = arr[0], arr[arr.size // 2], arr[arr.size - 1]
    
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:  # (a >= b)
        if a < c:
            return a
        else:
            return c
    else:  # (a >= b) and (c >= b)
        return b


@numba.njit
def quicksort_partition_hoare(x, y, lo, hi):
    """lo and hi are inclusive."""
    pivot = median_of_3(x[lo:hi + 1])
    
    # re-use lo and hi as moving indices
    lo -= 1
    hi += 1
    
    while True:
        lo += 1
        while x[lo] <= pivot:
            lo += 1
        
        hi -= 1
        while x[hi] >= pivot:
            hi -= 1
        
        if lo >= hi:
            return hi
        
        x[lo], x[hi] = x[hi], x[lo]
        y[lo], y[hi] = y[hi], y[lo]


@numba.njit
def max_heapify(x, y, index, n):
    # n = len(heap)
    while True:
        l = (index + 1) * 2 - 1
        
        if l >= n:
            break
        
        r = (index + 1) * 2
        
        if x[l] > x[index]:
            
            if r < n and x[r] > x[l]:
                x[r], x[index] = x[index], x[r]
                y[r], y[index] = y[index], y[r]
                index = r
            
            else:
                x[l], x[index] = x[index], x[l]
                y[l], y[index] = y[index], y[l]
                index = l
        
        elif r < n and x[r] > x[index]:
            x[r], x[index] = x[index], x[r]
            y[r], y[index] = y[index], y[r]
            index = r
        
        else:
            break


@numba.njit
def build_max_heap(x, y, n):
    for i in range(len(x) // 2 + 2, -1, -1):
        max_heapify(x, y, i, n)


@numba.njit
def heapsort(x, y):
    n = x.size
    # Build a max heap
    build_max_heap(x, y, n)
    
    # Sequentially put maximum values at end of array
    for i in range(x.size - 1, 0, -1):
        x[0], x[i] = x[i], x[0]
        y[0], y[i] = y[i], y[0]
        max_heapify(x, y, 0, i)


@numba.njit
def enlarge_first_dimension(x, new_size=-1):
    if new_size < 0:
        new_size = x.shape[0] * 2
    new_shape = (new_size, ) + x.shape[1:]
    ret = np.empty(new_shape, dtype=x.dtype)
    ret[:x.shape[0]] = x
    return ret


@numba.njit
def is_sorted(x):
    if x.size <= 1:
        return True
    for i in range(x.size - 1):
        if x[i] > x[i + 1]:
            return False
    return True


@numba.njit
def introsort_recursive(x, y, lo, hi, depth):
    """Sort x and y in place based on the values in x.

    Uses Quicksort for well-behaved arrays. If the stack depth grows too fast
    Heapsort is used. Insertion Sort is used for sub-arrays below a certain
    size."""
    tmp_x = x[0]
    tmp_y = y[0]
    
    while hi > lo:
        
        # Insertion sort on small sub-arrays
        if hi - lo < 24:
            
            for i in range(lo + 1, hi + 1):
                tmp_x = x[i]
                tmp_y = y[i]
                j = i - 1
                while j >= lo and x[j] > tmp_x:
                    x[j + 1] = x[j]
                    y[j + 1] = y[j]
                    j -= 1
                
                x[j + 1] = tmp_x
                y[j + 1] = tmp_y
            
            return
        
        # Heapsort if stack depth exceeds arbitrary limit
        if depth <= 0:
            heapsort(x[lo:hi + 1], y[lo:hi + 1])
            return
        
        depth -= 1
        
        # Hoare Partitioning
        pivot = median_of_3(x[lo: hi + 1])
        l = lo - 1
        r = hi + 1
        
        while True:
            l += 1
            while l < hi and x[l] <= pivot:
                l += 1
            
            r -= 1
            while r > lo and x[r] > pivot:
                r -= 1
            
            if l >= r:
                break
            
            tmp_x = x[r]
            x[r] = x[l]
            x[l] = tmp_x
            
            tmp_y = y[r]
            y[r] = y[l]
            y[l] = tmp_y
        
        introsort_recursive(x, y, lo, r, depth)
        lo = r


@numba.njit
def introsort(x, y):
    max_d = int(2 * np.log(x.size)) + 1
    introsort_recursive(x, y, 0, x.size - 1, max_d)