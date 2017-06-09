import numpy as np
import numba
import collections

from .utils import enlarge_first_dimension, mean_and_sse
from .partition import best_partition


spec = collections.OrderedDict()
spec["max_depth"] = numba.int32
spec["min_samples_leaf"] = numba.int32
spec["max_features"] = numba.int32
spec["min_samples_split"] = numba.int32
@numba.jitclass(spec)
class TreeParams:
    def __init__(self):
        pass


# stack columns
PARENT = np.int32(0)
START = np.int32(1)
END = np.int32(2)

n_integral_stack_columns = END + 1

# stack_f columns
NODE_MEAN = np.int32(0)
NODE_IMPURITY = np.int32(1)
RIGHT_CHILD_MEAN = np.int32(2)
RIGHT_CHILD_IMPURITY = np.int32(3)

# n_stack_columns = RIGHT_CHILD_IMPURITY + 1
n_stack_columns = END + 2 + RIGHT_CHILD_IMPURITY


@numba.njit
def build_tree(X, y, params, how, random_seed):
    """Build a regression tree without recursion.
    
    The tree is stored in a flattened 2d array."""
    
    # The shape of the data to
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
    
    parent_end = n
    
    est_samples_per_leaf = max(MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT // 2)
    
    # Stack is an array that is treated like a FIFO data-structure
    if params.max_depth > 0:
        stack_size = params.max_depth + 1
    else:
        stack_size = np.int32(
            np.ceil(np.log2(np.ceil(n // est_samples_per_leaf) * 2)))
    
    stack_i = np.empty((stack_size, n_stack_columns), dtype=np.int32)
    stack_f = stack_i[:, n_integral_stack_columns:].view(np.float32)
    stack_f[:] = np.nan
    stack_index = np.int32(0)  # Stack must starts with one element
    
    stack_i[stack_index, PARENT] = -1
    stack_i[stack_index, START] = 0
    stack_i[stack_index, END] = n
    tmp = mean_and_sse(y)
    stack_f[stack_index, NODE_MEAN] = tmp[0]
    stack_f[stack_index, NODE_IMPURITY] = tmp[1]
    stack_f[stack_index, RIGHT_CHILD_MEAN] = np.nan
    stack_f[stack_index, RIGHT_CHILD_IMPURITY] = np.nan
    
    # The current tree node
    node = np.int32(0)
    
    # Raw backing memory for contiguous access
    raw_array_size = 1024
    raw_array = np.empty((raw_array_size, 6), np.int32)
    dim__n_node_samples__left__right = raw_array[:, 0:4]
    decision_est__impurity = raw_array[:, 4:].view(np.float32)
    decision_est__impurity[0, 1] = tmp[1]
    
    contrib_data = np.uint32(0)
    
    best_part_result = np.empty(4, np.float32)
    temp_part_result = np.empty(4, np.float32)
    
    for node in range(n * 2):
        
        # Resize the output array if it is too large to
        if node >= raw_array_size:
            raw_array_size = min(raw_array_size * 2, n * 2)
            raw_array = enlarge_first_dimension(raw_array, raw_array_size)
            dim__n_node_samples__left__right = raw_array[:, 0:4]
            decision_est__impurity = raw_array[:, 4:].view(np.float32)
        
        # Each iteration deals with the portion X[start:end, :], y[start:end]
        # Either splitting it up further or setting it to a terminal leaf
        start = stack_i[stack_index, START]
        end = stack_i[stack_index, END]
        m = end - start
        
        # These values are computed and set by the parent before this node reached
        node_mean = stack_f[stack_index, NODE_MEAN]  # Mean of y[start:end]
        node_impurity = stack_f[
            stack_index, NODE_IMPURITY]  # Impurity of y[start:end]
        
        # This is a leaf node: add nothing to the stack
        if (m < MIN_SAMPLES_LEAF * 2
            or m < MIN_SAMPLES_SPLIT
            or stack_index >= MAX_DEPTH):
            dim__n_node_samples__left__right[node, 0] = -1
            dim__n_node_samples__left__right[node, 1] = m
            dim__n_node_samples__left__right[node, 2] = -1
            dim__n_node_samples__left__right[node, 3] = -1
            decision_est__impurity[node, 0] = node_mean
            decision_est__impurity[node, 1] = node_impurity
            contrib_data += m
        
        # This is not a terminal node, we partition it
        else:
            _dim, split_index = best_partition(
                X[start:end], y[start:end], MAX_FEATURES, MIN_SAMPLES_LEAF,
                dims_permutation, how, node_mean, node_impurity,
                best_part_result, temp_part_result)
            
            # The criterion is middway between the two cut points.
            dim__n_node_samples__left__right[node, 0] = _dim
            dim__n_node_samples__left__right[node, 1] = m
            dim__n_node_samples__left__right[node, 2] = node + 1
            # dim__n_node_samples__left__right[node, 3] = -1
            decision_est__impurity[node, 0] = X[start + split_index, _dim]
            decision_est__impurity[node, 1] = node_impurity
            
            stack_f[stack_index, RIGHT_CHILD_MEAN] = best_part_result[2]
            stack_f[stack_index, RIGHT_CHILD_IMPURITY] = best_part_result[3]
            
            stack_index += 1
            # Now we're referring to the child.
            
            if stack_index >= stack_i.shape[0]:
                stack_i = enlarge_first_dimension(stack_i,
                                                  stack_i.shape[0] * 2)
                stack_f = stack_i[:, n_integral_stack_columns:].view(
                    np.float32)
            
            # Set the parent node of the child to the current node
            stack_i[stack_index, PARENT] = node
            stack_i[stack_index, START] = start
            stack_i[stack_index, END] = start + split_index
            stack_f[stack_index, NODE_MEAN] = best_part_result[
                0]  # .left_mean
            stack_f[stack_index, NODE_IMPURITY] = best_part_result[
                1]  # .left_sse
            continue
        
        # 'Unwind' the stack until either:
        #  (1) We have no more stack elements
        #  (2) We can add a right node to one of the elements:
        #      in this case `end` is not equal to `parent_end`.
        current_node = node
        parent_node = stack_i[stack_index, PARENT]
        parent_end = stack_i[stack_index - 1, END]
        while stack_index > 0 and end == parent_end:
            dim__n_node_samples__left__right[parent_node, 3] = current_node
            stack_index -= 1
            current_node = parent_node
            end = parent_end
            parent_node = stack_i[stack_index, PARENT]
            parent_end = stack_i[stack_index - 1, END]
        
        # Now, either we're finished building the tree...
        if stack_index <= 0:
            break
        
        # Or we add the right counterpart of the left node we've unwound to
        if stack_index >= 1:  # and end != stack[stack_index - 1,]:
            # Parent of a right node is the same as the left node
            #     stack_i[stack_index, PARENT] = <same>  <- Not needed
            stack_i[stack_index, START] = end
            stack_i[stack_index, END] = parent_end  # The parent's slice end
            stack_f[stack_index, NODE_MEAN] = stack_f[
                stack_index - 1, RIGHT_CHILD_MEAN]
            stack_f[stack_index, NODE_IMPURITY] = stack_f[
                stack_index - 1, RIGHT_CHILD_IMPURITY]
            #stack_f[stack_index, PARENT_SPLIT_FEATURE]
    
    assert contrib_data == n  # We should have touched all of the `y` data once
    return node + 1, raw_array[:node + 1].copy()


@numba.njit
def _predict(flat_tree, X_pred):
    """Predict values from a regression tree."""
    n, n_features = X_pred.shape
    # dims = flat_tree[:, 0]
    threshold_or_est = flat_tree[:, 4].view(numba.float32)
    
    # dim__n_node_samples__left__right = raw_array[:, 0:4]
    # decision_est__impurity = raw_array[:, 4:].view(np.float32)
    
    y_pred = np.empty(n, np.float32)
    
    dim = np.int32(0)
    node = np.int32(0)
    
    for i in range(n):
        node = 0
        dim = flat_tree[node, 0]
        while dim >= 0:
            if X_pred[i, dim] < threshold_or_est[node]:
                node = flat_tree[node, 2]
            else:
                node = flat_tree[node, 3]
            dim = flat_tree[node, 0]
        
        y_pred[i] = threshold_or_est[node]
    
    return y_pred


@numba.njit
def _feature_importances(flat_tree, n_features):
    """Calculate the proxy feature importance as the average decrease in impurity."""
    n = flat_tree.shape[0]
    dim_or_size = flat_tree[:, 0]
    # n_node_samples = flat_tree[:, 1]
    left = flat_tree[:, 2]
    right = flat_tree[:, 3]
    #crit_or_est = flat_tree[:, 4].view(np.float32)
    impurity = flat_tree[:, 5]
    
    sum_impurity = np.zeros(n_features)
    for node in range(n):
        if left[node] > 0:  # Node is not a leaf
            impurity[node]
            diff = impurity[node] - impurity[left[node]] - impurity[
                right[node]]
            sum_impurity[dim_or_size[node]] += diff
    return sum_impurity / sum_impurity.max()