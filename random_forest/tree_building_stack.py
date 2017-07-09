import numba
from collections import OrderedDict

linked_node_spec = OrderedDict()

node_type = numba.deferred_type()

linked_node_spec["parent"] = node_type
linked_node_spec["child"] = node_type

linked_node_spec["depth"] = numba.int32
linked_node_spec["start"] = numba.int32
linked_node_spec["end"] = numba.int32
linked_node_spec["index"] = numba.int32

linked_node_spec["node_mean"] = numba.float32
linked_node_spec["node_impurity"] = numba.float32
linked_node_spec["right_child_mean"] = numba.float32
linked_node_spec["right_child_impurity"] = numba.float32

linked_node_spec["sorted_dim"] = numba.int32

@numba.jitclass(linked_node_spec)
class LinkedNode:
    def __init__(self, depth):
        self.parent = self
        self.child = self
        self.depth = depth


node_type.define(LinkedNode.class_type.instance_type)


@numba.njit
def Stack():
    return LinkedNode(0)


@numba.njit
def same(a, b):
    return a.depth == b.depth


@numba.njit
def is_root(node):
    return node.depth == node.parent.depth


@numba.njit
def is_top(node):
    return node.depth == node.child.depth


@numba.njit  # ( node_type ( node_type ) )
def push(stack):
    """Append (or re-use) an element to the right of the stack."""
    if not is_top(stack):
        return stack.child
    
    else:
        l = LinkedNode(stack.depth + 1)
        stack.child = l
        l.parent = stack
        return l


@numba.njit
def is_right_child(node):
    return node.depth >= 1 and node.end == node.parent.end


@numba.njit
def is_left_child(node):
    return node.depth >= 1 and node.start == node.parent.start


@numba.njit
def unwind_stack(stack, RIGHT):
    # RIGHT[stack.parent.index] = stack.index
    
    # while is_right_child(stack):
    while is_right_child(stack):
        RIGHT[stack.parent.index] = stack.index
        # RIGHT[stack.parent.index] = stack.depth
        stack = stack.parent
    
    # assert stack.depth == 0 or is_left_child(stack)
    # assert is_left_child(stack.child)
    return stack


@numba.njit
def push_start_end(stack, start, end):
    r = push(stack)
    r.start = start
    r.end = end
    return r


@numba.njit
def replace_left_node_with_right(stack):
    # assert not is_root(stack)
    # print('SWAPPING ***')
    stack.start = stack.end
    stack.end = stack.parent.end
    stack.node_mean = stack.parent.right_child_mean
    stack.node_impurity = stack.parent.right_child_impurity