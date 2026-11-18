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
import numpy as np

class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype

    def __array__(self):
        return np.asarray(self._np, dtype=dtype)
    
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

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    x = self._np
    y = other._np
    if op.name == 'ADD':
        out = x + y
    elif op.name == 'SUB':
        out = x - y
    elif op.name == 'MUL':
        out = x * y
    elif op.name == 'DIV':
        out = x / y
    elif op.name == 'CMPLT':
        out = (x < y).astype(x.dtype)
    elif op.name == 'MAX':
        out = np.maximum(x, y)
    else:
        raise ValueError(f"Invalid operation: {op.name}")
    
    return LazyBuffer(out)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    name = getattr(op, 'name', None)
    if name == 'SUM': out = self._np.sum(axis=axis, keepdims=True)
    elif name == 'MAX': out = self._np.max(axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce op: {name}")
    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    new_shape = tuple(int(d) for d in new_shape)

    return LazyBuffer(np.array(np.broadcast_to(self._np, new_shape)))

# Step 12 - lazybuffer_permute
def permute(self, order):
    return LazyBuffer(self._np.transpose(order))

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        flags = [t.requires_grad for t in tensors]
        self.needs_input_grad = flags
        if any(f is True for f in flags):
            self.requires_grad = True
        elif None in flags:
            self.requires_grad = None
        else:
            self.requires_grad = False

        if self.requires_grad is True:
            self.parents = tensors

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")
        
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")
        
    Function.forward = forward
    Function.backward = backward
    
    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    ctx = cls(*tensors)
    bufs = [t.lazydata for t in tensors]
    out_buf = ctx.forward(*bufs, **kwargs)
    result = Tensor(out_buf, requires_grad=ctx.requires_grad)

    if ctx.requires_grad is True:
        result._ctx = ctx
    return result


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        UnaryOps, *_ = make_op_enums()
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, *_ = make_op_enums()
        zero = LazyBuffer.const(0, self.ret.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, mask)

# Step 18 - Log
class Log(Function):
    def forward(self, x):
        self.x = x
        UnaryOps, *_ = make_op_enums()
        self.ret = x.e(UnaryOps.LOG)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, *_ = make_op_enums()
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.x)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x):
        self.x = x
        UnaryOps, *_ = make_op_enums()
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, *_ = make_op_enums()
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        self.x = x
        UnaryOps, *_ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, *_ = make_op_enums()
        two = LazyBuffer(2)
        den = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, den)

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        self.x = x
        UnaryOps, *_ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        ones = LazyBuffer.const(1.0, self.ret._np.shape)
        one_minus_y = lazybuffer_binary_e(ones, BinaryOps.SUB, self.ret)
        sig_deriv = lazybuffer_binary_e(self.ret, BinaryOps.MUL, one_minus_y)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, sig_deriv)

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)


    def backward(self, grad_output):
        # TODO: route grad_output to each input that requires a gradient
        return (
            grad_output if self.needs_input_grad[0] else None,
            grad_output if self.needs_input_grad[1] else None,
        )

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)


    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()
        gx = g if self.needs_input_grad[0] else None
        gy = g.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        return (gx, gy)

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y

        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        _, BinaryOps, _, _ = make_op_enums()
        gx = lazybuffer_binary_e(self.y, BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None
        gy = lazybuffer_binary_e(self.x, BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None
        return (gx, gy)

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y

        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        grad_x = None
        grad_y = None

        if self.needs_input_grad[0] is True:
            grad_x = LazyBuffer(grad_output._np / self.y._np)

        if self.needs_input_grad[1] is True:
            grad_y = LazyBuffer(-grad_output._np * self.x._np / (self.y._np ** 2))

        return (grad_x, grad_y)

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        result = x._np.sum(axis=axis, keepdims=True)
        return LazyBuffer(result)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    g = grad_output._np
    g = np.broadcast_to(g, self.input_shape)
    g = np.ascontiguousarray(g)
    return LazyBuffer(g)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        self.x = x
        self.axis = axis
        out = x._np.max(axis=axis, keepdims=True)
        self.ret = LazyBuffer(out)
        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    _, BinaryOps, ReduceOps, _ = make_op_enums()
    
    x_shape = self.x._np.shape
    ret_expanded = LazyBuffer(np.broadcast_to(self.ret._np, x_shape))

    ones = LazyBuffer(np.ones(x_shape, dtype=self.x.dtype))
    cmplt = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, ret_expanded)
    is_max = lazybuffer_binary_e(ones, BinaryOps.SUB, cmplt)
    
    counts_raw = np.sum(is_max._np, axis=self.axis, keepdims=True)
    counts = LazyBuffer(np.broadcast_to(counts_raw, x_shape))

    mask_normalized = lazybuffer_binary_e(is_max, BinaryOps.DIV, counts)
    grad_expanded = LazyBuffer(np.broadcast_to(grad_output._np, x_shape))
    
    return lazybuffer_binary_e(mask_normalized, BinaryOps.MUL, grad_expanded)

Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return LazyBuffer(x._np.reshape(shape))

    def backward(self, grad_output):
        return LazyBuffer(grad_output._np.reshape(self.input_shape))

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    ctx.input_shape = x.shape
    new_shape = tuple(int(d) for d in shape)
    out = np.broadcast_to(x._np, new_shape)
    return LazyBuffer(np.ascontiguousarray(out))

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    input_shape = ctx.input_shape
    grad_shape = grad_output.shape
    
    # 1. Find the axes that were broadcasted (size was 1, now larger)
    axes = tuple(i for i in range(len(input_shape)) 
                 if input_shape[i] == 1 and grad_shape[i] != 1)
    
    # 2. Perform the reduction using numpy on the underlying buffer array, keeping dimensions
    np_grad = grad_output._np
    if axes:
        np_grad = np_grad.sum(axis=axes, keepdims=True)
        
    # 3. Reshape back to the exact input shape and wrap in a LazyBuffer
    return LazyBuffer(np_grad.reshape(input_shape))

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    def forward(ctx, x, order):
        ctx.order = order
        return permute(x, order)

    def backward(ctx, grad_output):
        inv = argsort(ctx.order)
        return permute(grad_output, inv)

    return (forward, backward)

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

