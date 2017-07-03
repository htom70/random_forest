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
    
    # re-use lo and hi to correspond to moving indices
    lo -= 1
    hi += 1
    
    while True:
        lo += 1
        while x[lo] < pivot:
            lo += 1
        
        hi -= 1
        while x[hi] > pivot:
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
        r = (index + 1) * 2
        
        if l >= n:
            break
        
        elif x[l] > x[index]:
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
        else:
            break


@numba.njit
def heapsort(x, y):
    # Build a max heap
    for i in range(len(x) // 2, -1, -1):
        max_heapify(x, y, i, x.size)
    
    # Sequentially put maximum values at end of array
    for i in range(x.size - 1, -1, -1):
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
def introsort(x, y):
    """Quicksort without recursion."""
    stack_size = max(int(np.log2(x.size)), 1)
    
    recursion_limit = int(2 * np.log(x.size))
    
    lo_hi_stack = np.empty((stack_size, 2), np.int64)
    
    top = np.uint32(0)
    lo_hi_stack[top, 0] = 0
    lo_hi_stack[top, 1] = x.size - 1
    
    while True:
        lo = lo_hi_stack[top, 0]
        hi = lo_hi_stack[top, 1]
        
        if hi - lo < 32:
            insertion_sort(x, y, lo, hi)
        
        elif top > recursion_limit:
            # If the stack height exceeds the recursion limit sort this chunk
            # with heapsort
            heapsort(x[lo:hi + 1], y[lo:hi + 1])
        
        else:
            # p = quicksort_partition_lomuto(x, lo, hi)
            p = quicksort_partition_hoare(x, y, lo, hi)
            top += 1
            
            if stack_size <= top:  # Resize the stack if we need more room
                stack_size *= 2
                lo_hi_stack = enlarge_first_dimension(lo_hi_stack, stack_size)
            
            lo_hi_stack[top, 0] = lo
            lo_hi_stack[top, 1] = p
            
            continue
        
        if top == 0:
            return
        
        # Unwind the stack
        while lo_hi_stack[top, 1] == lo_hi_stack[top - 1, 1]:
            top -= 1
            if top == 0:
                return
        
        # Now add the upper counterpart of the sub-array indices we unwound to
        lo_hi_stack[top, 0] = lo_hi_stack[top, 1] + 1
        lo_hi_stack[top, 1] = lo_hi_stack[top - 1, 1]