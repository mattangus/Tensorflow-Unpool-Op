"""
Module that provides access to the custom unpool op c++ implementation
"""
import tensorflow as tf
from tensorflow.python.framework import ops

_unpool_op_module = tf.load_op_library('./libunpool_op.so')


def unpool(x, inds, output_size, **kwargs):
    """
    Wrapper function for the unpool operation.
    This operation takes an input tensor, x, and a set
    of indices, inds. The indices map values from x to
    a tensor with the same shape as the tensor before a
    pooling operation was applied.

    :param x: input tensor, values to be mapped
    :param inds: tensor with indices used in mapping
    :param output_size: the size of the output tensor (must be length 4) 
    :param kwargs: any remaining parameters to pass to tensorflow
    (e.g. name)
    """
    return _unpool_op_module.unpool(x, inds, output_size=output_size,
                                    **kwargs)


def unpool_grad(inds, grad, **kwargs):
    """
    Wrapper function for the unpool gradient operation.
    This operation takes an input tensor, x, a set
    of indices, inds, and a the gradient. The indices map
    values from grad to a tensor with the same shape as the
    tensor before a pooling operation was applied.

    :param x: input tensor, values to be mapped
    :param inds: tensor with indices used in mapping
    :param grad: tensor containing the gradient
    :param kwargs: any remaining parameters to pass to tensorflow
    (e.g. name)
    """
    return _unpool_op_module.unpool_grad(inds, grad, **kwargs)


@ops.RegisterGradient("Unpool")
def _unpool_grad(op, grad):
    """
    Tensorflow wrapper for the gradient function.
    This is called when tf.gradients is called, then
    the unpool_grad is added to the graph.

    :param op: the original unpool op
    :param grad: the gradient being back propagated
    """
    return unpool_grad(op.inputs[1], grad)
