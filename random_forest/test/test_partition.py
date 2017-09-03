import numpy as np
import numba
from ..partition import PartitionResult, _best_partition

N_TESTS = 10000
MAX_TEST_ARRAY_SIZE = 1000


def gen_array_size():
    return np.random.randint(2, MAX_TEST_ARRAY_SIZE)


#@numba.njit
def sse(x):
    return np.var(x) * x.size


#@numba.njit
def min_sse_reference(x, min_samples_leaf=1):
    assert x.size >= 2
    min_sse = np.inf
    min_index = 0
    for i in range(min_samples_leaf, x.size - min_samples_leaf + 1):
        _sse = sse(x[:i]) + sse(x[i:])
        #print(_sse)
        if _sse < min_sse:
            min_index, min_sse = i, _sse
    return min_index


def test_partition_randomized():
    pr = PartitionResult()

    min_samples_leaf = 1
    
    for _ in range(N_TESTS):
        
        x = np.random.rand(gen_array_size())  # + np.arange(6)
        _best_partition(x, pr, min_samples_leaf)
        ref = min_sse_reference(x, min_samples_leaf)
        assert ref == pr.index