import numpy as np


def expand2d(x):
    """
    Expand the dimensions to 4d if the input is 2d.
    Otherwise keep it the same.

    :param x: array to expand
    :return: nd array where n = 4 if dim(x) = 2, n = dim(x) otherwise.
    """
    if len(x.shape) == 2:
        return np.expand_dims(np.expand_dims(x, axis=0), axis=-1)
    else:
        return x


class UnpoolExample(object):
    """
    Container class for examples to test for the unpool operation.
    """

    def __init__(self, x, inds, output_size, result, grad, grad_exp):
        self.x = expand2d(x)
        self.inds = expand2d(inds)
        self.output_size = output_size
        self.result = expand2d(result)
        self.grad = expand2d(grad)
        self.grad_exp = expand2d(grad_exp)
