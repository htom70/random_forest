import numpy as np
import numba
from .utils import enlarge_first_dimension, mean_and_sse
from .partition import best_partition, PartitionResult
from .tree_building_stack import *


spec = OrderedDict()
spec["max_depth"] = numba.int32
spec["min_samples_leaf"] = numba.int32
spec["max_features"] = numba.int32
spec["min_samples_split"] = numba.int32
@numba.jitclass(spec)
class TreeParams:
    def __init__(self):
        pass


# Flattening tree structure containing the decisions.
FLAT_TREE_COLUMNS = 8

@numba.njit
def gen_flat_tree(size):
    raw_tree = np.empty((size, FLAT_TREE_COLUMNS), dtype=np.int32)
    return raw_tree


@numba.njit
def flat_tree_fields(flat_tree):
    """(DIM, N, LEFT, RIGHT, DECISION, MEAN, IMPURITY)"""
    return (flat_tree[:, 0],
            flat_tree[:, 1],
            flat_tree[:, 2],
            flat_tree[:, 3],
            flat_tree[:, 5].view(np.float32),
            flat_tree[:, 6].view(np.float32),
            flat_tree[:, 7].view(np.float32))


@numba.njit
def grow_flat_tree(raw_tree):
    """FLAT_TREE, (DIM, N, LEFT, RIGHT, PARENT, DECISION, MEAN, IMPURITY)"""
    size = raw_tree.shape[0] * 2
    bigger = np.empty((size, FLAT_TREE_COLUMNS), dtype=np.int32)
    bigger[:raw_tree.shape[0]] = raw_tree
    return bigger
    # return bigger, flat_tree_fields(raw_tree)


@numba.njit
def build_tree(X, y, params, how, random_seed):
    n, n_features = X.shape
    
    MAX_DEPTH = params.max_depth
    if MAX_DEPTH <= 0:
        MAX_DEPTH = n * 2
    MAX_DEPTH = min(n * 2, MAX_DEPTH)
    MAX_FEATURES = params.max_features
    MIN_SAMPLES_SPLIT = params.min_samples_split
    MIN_SAMPLES_LEAF = params.min_samples_leaf
    
    dims_permutation = np.empty(n_features, np.int32)
    for i in range(n_features):
        dims_permutation[i] = i
    
    np.random.seed(random_seed)
    
    node_mean, node_impurity = mean_and_sse(y)
    
    stack = Stack()
    stack.start = 0
    stack.end = n
    stack.index = 0
    stack.node_mean = node_mean
    stack.node_impurity = node_impurity
    
    flat_tree = gen_flat_tree(1024)
    (DIM, N_SAMPLES, LEFT, RIGHT, DECISION, NODE_MEAN,
     NODE_IMPURITY) = flat_tree_fields(flat_tree)
    
    DECISION[:] = np.nan
    NODE_MEAN[:] = np.nan
    NODE_IMPURITY[:] = np.nan
    
    DIM[:] = -1
    N_SAMPLES[:] = -1
    LEFT[:] = -1
    RIGHT[:] = -1
    
    temp_a = PartitionResult()
    temp_b = PartitionResult()
    
    last_best_dim = -1
    
    node = -1
    
    while True:
        
        node += 1
        # print('NODE', node, '-----------------------------------------')
        # print_stack(stack)
        
        stack.index = node
        
        if node >= DIM.size:
            flat_tree = grow_flat_tree(flat_tree)
            (DIM, N_SAMPLES, LEFT, RIGHT, DECISION,
             NODE_MEAN, NODE_IMPURITY) = flat_tree_fields(flat_tree)
        
        start, end = stack.start, stack.end
        m = end - start
        
        # print('node mean', stack.node_mean, np.mean(y[start:end]))
        # print('node impurity', stack.node_impurity, sse(y[start:end]))
        # assert_close(stack.node_mean, np.mean(y[start:end]), 1e-2)
        # assert_close(stack.node_impurity, sse(y[start:end]), 1e-2)
        
        N_SAMPLES[node] = m
        node_mean = stack.node_mean
        NODE_MEAN[node] = node_mean
        node_impurity = stack.node_impurity
        NODE_IMPURITY[node] = node_impurity
        
        # If the node is a leaf node then write then write to permanent
        # flat_tree and unwind
        if (m <= MIN_SAMPLES_LEAF * 2
            or m <= MIN_SAMPLES_SPLIT
            or stack.depth > MAX_DEPTH):
            
            N_SAMPLES[node] = m
            DIM[node] = -2
            LEFT[node] = -1
            RIGHT[node] = -1
            DECISION[node] = np.nan
            
            # Unwind the stack until we have found a left node
            # This sets the RIGHT_CHILD_{MEAN, IMPURITY} fields
            # in the flat tree, something we have not yet done.
            stack = unwind_stack(stack, RIGHT)
            
            if is_root(stack):
                break  # If this is the root node we're done
            else:
                # Otherwise we switch to the right counterpart of the
                # left node we've unwound to.
                assert is_left_child(stack)
                replace_left_node_with_right(stack)
                
                # print_stack(stack)
        
        else:
            # Partition X[start:end], y[start:end] based on the
            # best predictor.
            best = best_partition(X[start:end], y[start:end],
                                  MAX_FEATURES, MIN_SAMPLES_LEAF,
                                  dims_permutation, how,
                                  node_mean, node_impurity, temp_a, temp_b)
            
            last_best_dim = best.dim
            # assert best.index >= MIN_SAMPLES_LEAF
            
            # Set RIGHT_CHILD_MEAN and RIGHT_CHILD_IMPURITY for current node
            stack.right_child_mean = best.right_mean
            stack.right_child_impurity = best.right_sse
            
            # Flat tree elements
            DIM[node] = best.dim
            DECISION[node] = best.split_value
            LEFT[node] = node + 1
            
            # Now add the info for the next node
            stack = push(stack)
            stack.start = start
            stack.end = start + best.index
            stack.node_mean = best.left_mean
            stack.node_impurity = best.left_sse
            if best.left_is_sorted:
                stack.sorted_dim = best.dim
            # stack.index = node + 1
    
    return flat_tree[:node + 1]


@numba.njit
def _predict(flat_tree, X_pred):
    (DIMENSION, N_SAMPLES, LEFT, RIGHT, DECISION,
     NODE_MEAN, NODE_IMPURITY) = flat_tree_fields(flat_tree)
    
    n, n_features = X_pred.shape
    y_pred = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        node = 0
        dim = DIMENSION[0]
        while dim >= 0:
            if X_pred[numba.int32(i), dim] <= DECISION[node]:
                node = LEFT[node]
            else:
                node = RIGHT[node]
            dim = DIMENSION[node]
        y_pred[i] = NODE_MEAN[node]
    return y_pred


# @numba.njit
# def _feature_importances(flat_tree, n_features):
#     """Calculate the proxy feature importance as the average decrease in impurity."""
#     n = flat_tree.shape[0]
#     dim_or_size = flat_tree[:, 0]
#     # n_node_samples = flat_tree[:, 1]
#     left = flat_tree[:, 2]
#     right = flat_tree[:, 3]
#     # crit_or_est = flat_tree[:, 4].view(np.float32)
#     impurity = flat_tree[:, 5]
#
#     sum_impurity = np.zeros(n_features)
#     for node in range(n):
#         if left[node] > 0:  # Node is not a leaf
#             impurity[node]
#             diff = impurity[node] - impurity[left[node]] - impurity[
#                 right[node]]
#             sum_impurity[dim_or_size[node]] += diff
#     return sum_impurity / sum_impurity.max()