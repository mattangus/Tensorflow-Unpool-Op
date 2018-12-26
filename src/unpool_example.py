"""
Example script using the unpool op. It is very similar to tf ops.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import numpy as np
import unpool


def print_tensor(x, shape):
    """
    helper function to print 4d tensors
    """
    for p in range(shape[0]):
        for q in range(shape[-1]):
            print(p, q)
            print(x[p, :, :, q])

def main():
    # set up shape inputs
    k = 2
    height = 4 * k
    width = 4 * k
    examples = 2
    channels = 2

    # initialize input values
    vals = np.reshape(np.arange(
        examples * height * width * channels, dtype=np.float32),
        newshape=(examples, height, width, channels))
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]

    # set up layers
    X = tf.placeholder(shape=vals.shape, name="x", dtype="float32")
    pool, inds = tf.nn.max_pool_with_argmax(X, ksize=ksize,
                                            strides=strides,
                                            padding="VALID",
                                            name="pool")
    unpool_layer = unpool.unpool(pool, inds,
                                 output_size=[height, width],
                                 name="unpool")

    # make sure we use the gradient function
    # implemented for the unpool op
    base_grad = tf.gradients(unpool_layer, pool)

    # run and print each element of the setup
    with tf.Session() as sess:
        feed = {X: vals}

        print("vals")
        print_tensor(vals, vals.shape)

        print("pool")
        pool_val = sess.run(pool, feed_dict=feed)
        print_tensor(pool_val, pool_val.shape)

        print("inds")
        inds_val = sess.run(inds, feed_dict=feed)
        print_tensor(inds_val, inds_val.shape)

        print("unpool")
        unpool_val = sess.run(unpool_layer, feed_dict=feed)
        print_tensor(unpool_val, unpool_val.shape)

        grad = sess.run(base_grad, feed_dict=feed)
        print("unpool grad")
        GRAD_VAL = grad[0]
        print_tensor(GRAD_VAL, GRAD_VAL.shape)


if __name__ == "__main__":
    main()
