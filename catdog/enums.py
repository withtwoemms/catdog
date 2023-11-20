from enum import auto, Enum


class Activations(Enum):
    deserialize = auto()
    elu = auto()
    exponential = auto()
    gelu = auto()
    get = auto()
    hard_sigmoid = auto()
    linear = auto()
    mish = auto()
    relu = auto()
    selu = auto()
    serialize = auto()
    sigmoid = auto()
    softmax = auto()
    softplus = auto()
    softsign = auto()
    swish = auto()
    tanh = auto()
