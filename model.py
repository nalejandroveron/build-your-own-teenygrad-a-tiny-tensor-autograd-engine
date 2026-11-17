"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result

# Step 2 - argsort
from functools import cmp_to_key

def argsort(values):
    idx = list(range(len(values)))

    def compare(a, b):
        if values[a] < values[b]:
            return -1
        elif values[a] > values[b]:
            return 1
        return 0

    idx.sort(key=cmp_to_key(compare))

    return idx

# Step 3 - make_op_enums
import enum

def make_op_enums():
    unaryOps = enum.Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    binaryOps = enum.Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    reduceOps = enum.Enum('ReduceOps', ['SUM', 'MAX'])
    movementOps = enum.Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])

    return (unaryOps, binaryOps, reduceOps, movementOps)

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype

    def __array__(self):
        return np.asarray(self,_np, dtype=dtype)
    
    def __float__(self):
        float(self._np)

# Step 5 - lazybuffer_const
def const(value, shape):
    arr = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(arr)
LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    rng = np.random.RandomState(seed)
    data = rng.random(shape).astype(np.float32)
    return LazyBuffer(data)

# Step 7 - lazybuffer_unary_e
def e(self, op):
    x = self._np
    if op.name == 'NEG':
        out = -x
    elif op.name == 'RELU':
        out = np.maximum(x, 0)
    elif op.name == 'LOG':
        out = np.log(x)
    elif op.name == 'EXP':
        out = np.exp(x)
    elif op.name == 'SQRT':
        out = np.sqrt(x)
    elif op.name == 'SIGMOID':
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(f"Invalid operation: {op.name}")
    
    return LazyBuffer(out)


LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e (not yet solved)
# TODO: implement

# Step 9 - lazybuffer_r (not yet solved)
# TODO: implement

# Step 10 - lazybuffer_reshape (not yet solved)
# TODO: implement

# Step 11 - lazybuffer_expand (not yet solved)
# TODO: implement

# Step 12 - lazybuffer_permute (not yet solved)
# TODO: implement

# Step 13 - Function (not yet solved)
# TODO: implement

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

