import unittest
import numpy as np
import tensorflow as tf
import unpool
from unpool_test_utils import UnpoolExample


def get_valid_unpool1():
    """
    Get an unpool example where:
    batch = 1
    width = 2
    height = 2
    channels = 1
    out size = 4x4
    """
    x = np.array(
        [[1, 2],
         [3, 4]], dtype='float32')

    inds = np.array(
        [[5, 7],
         [13, 15]], dtype='int64')

    result = np.array(
        [[0, 0, 0, 0],
         [0, 1, 0, 2],
         [0, 0, 0, 0],
         [0, 3, 0, 4]], dtype='float32')

    grad = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]], dtype='float32')

    grad_exp = np.array(
        [[6, 8],
         [14, 16]], dtype='float32')

    output_size = [4, 4]

    return UnpoolExample(x, inds, output_size, result, grad, grad_exp)


def get_valid_unpool2():
    """
    Get an unpool example where:
    batch = 1
    width = 2
    height = 2
    channels = 1
    out size = 5x5
    """
    x = np.array(
        [[1, 2],
         [3, 4]], dtype='float32')

    inds = np.array(
        [[6, 8],
         [16, 18]], dtype='int64')

    result = np.array(
        [[0, 0, 0, 0, 0],
         [0, 1, 0, 2, 0],
         [0, 0, 0, 0, 0],
         [0, 3, 0, 4, 0],
         [0, 0, 0, 0, 0]], dtype='float32')

    grad = np.array(
        [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20],
         [21, 22, 23, 24, 25]], dtype='float32')

    grad_exp = np.array(
        [[7, 9],
         [17, 19]], dtype='float32')

    output_size = [5, 5]

    return UnpoolExample(x, inds, output_size, result, grad, grad_exp)


def get_unpool_with_channels():
    """
    Get an unpool example where:
    batch = 1
    width = 2
    height = 2
    channels = 2
    out size = 1x4x4x2
    """
    x = np.array(
        [[[[1, 2], [3, 4]],
          [[5, 6], [7, 8]]]], dtype='float32')

    inds = np.array(
        [[[[10, 11], [14, 15]],
          [[26, 27], [30, 31]]]], dtype='int64')

    result = np.array(
        [[[[0, 0], [0, 0], [0, 0], [0, 0]],
          [[0, 0], [1, 2], [0, 0], [3, 4]],
          [[0, 0], [0, 0], [0, 0], [0, 0]],
          [[0, 0], [5, 6], [0, 0], [7, 8]]]], dtype='float32')

    grad = np.array(
        [[[[1, 2], [3, 4], [5, 6], [7, 8]],
          [[9, 10], [11, 12], [13, 14], [15, 16]],
          [[17, 18], [19, 20], [21, 22], [23, 24]],
          [[25, 26], [27, 28], [29, 30], [31, 32]]]], dtype='float32')

    grad_exp = np.array(
        [[[[11, 12], [15, 16]],
          [[27, 28], [31, 32]]]], dtype='float32')

    output_size = [4, 4]

    return UnpoolExample(x, inds, output_size, result, grad, grad_exp)


def get_unpool_with_batches():
    """
    Get an unpool example where:
    batch = 2
    width = 2
    height = 2
    channels = 1
    out size = 2x5x5x1
    """
    x = np.array(
        [[[1, 2],
          [3, 4]],
         [[5, 6],
          [7, 8]]], dtype='float32')[..., None]

    inds = np.array(
        [[[6, 8],
          [16, 18]],
         [[6, 8],
          [16, 18]]], dtype='int64')[..., None]

    result = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 1, 0, 2, 0],
          [0, 0, 0, 0, 0],
          [0, 3, 0, 4, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 5, 0, 6, 0],
          [0, 0, 0, 0, 0],
          [0, 7, 0, 8, 0],
          [0, 0, 0, 0, 0]]], dtype='float32')[..., None]

    grad = np.array(
        [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
          [21, 22, 23, 24, 25]],
         [[26, 27, 28, 29, 30],
          [31, 32, 33, 34, 35],
          [36, 37, 38, 39, 40],
          [41, 42, 43, 44, 45],
          [46, 47, 48, 49, 50]]], dtype='float32')[..., None]

    grad_exp = np.array(
        [[[7, 9],
          [17, 19]],
         [[32, 34],
          [42, 44]]], dtype='float32')[..., None]

    output_size = [5, 5]

    return UnpoolExample(x, inds, output_size, result, grad, grad_exp)


def get_unpool_rect():
    """
    Get an unpool example where:
    batch = 1
    width = 4
    height = 2
    channels = 1
    out size = 4x8
    """
    x = np.array(
        [[[1, 2, 3, 4],
          [5, 6, 7, 8]]], dtype='float32')[..., None]

    inds = np.array(
        [[[9, 11, 13, 15],
          [25, 27, 29, 31]]], dtype='int64')[..., None]

    result = np.array(
        [[[0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 2, 0, 3, 0, 4],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 5, 0, 6, 0, 7, 0, 8]]], dtype='float32')[..., None]

    grad = np.array(
        [[[1, 2, 3, 4, 5, 6, 7, 8],
          [9, 10, 11, 12, 13, 14, 15, 16],
          [17, 18, 19, 20, 21, 22, 23, 24],
          [25, 26, 27, 28, 29, 30, 31, 32]]],
        dtype='float32')[..., None]

    grad_exp = np.array(
        [[[10, 12, 14, 16],
          [26, 28, 30, 32]]], dtype='float32')[..., None]

    output_size = [4, 8]

    return UnpoolExample(x, inds, output_size, result, grad, grad_exp)


valid_example_fns = [get_valid_unpool1, get_valid_unpool2,
                     get_unpool_with_channels,
                     get_unpool_with_batches, get_unpool_rect]


class UnpoolOpTest(unittest.TestCase):
    def test_validExamples(self):
        """
        Loop through all examples and check if the unpool is correct
        """
        with tf.Session():
            for example_fn in valid_example_fns:
                ex = example_fn()
                result = unpool.unpool(ex.x, ex.inds,
                                       output_size=ex.output_size)\
                    .eval()
                np.testing.assert_array_equal(ex.result, result)

    def test_validGrad(self):
        """
        Loop through all examples and check if the gradient is correct
        """
        with tf.Session() as sess:
            for example_fn in valid_example_fns:
                ex = example_fn()
                result = sess.run(
                    unpool.unpool_grad(ex.inds, ex.grad))[0]
                np.testing.assert_array_equal(ex.grad_exp, result)

    def test_bad_size(self):
        """
        Test a few bad output sizes
        """
        with tf.Session():
            with self.assertRaises(Exception):
                unpool.unpool([10., 11., 12.], [13., 14., 15.],
                              output_size=[1])
            with self.assertRaises(Exception):
                unpool.unpool([10., 11., 12.], [13., 14., 15.],
                              output_size=[1, 2, 2])
            with self.assertRaises(Exception):
                unpool.unpool([10., 11., 12.], [13., 14., 15.],
                              output_size=[1, 2, 2, 1])
            with self.assertRaises(Exception):
                unpool.unpool([10., 11., 12.], [13., 14., 15.],
                              output_size=[None, 2])

    def test_shape_inf(self):
        """
        Test that the shape inference works and graph build time
        """
        cases = [((None, 2, 2, 3), [4, 4], (None, 4, 4, 3)),
                 ((3, 2, 2, None), [4, 4], (3, 4, 4, None)),
                 ((None, 2, 2, None), [4, 4], (None, 4, 4, None)),
                 ((3, 2, 2, 5), [4, 4], (3, 4, 4, 5))]
        with tf.Session():
            for case in cases:
                x = tf.placeholder('float32', shape=case[0],
                                   name='test')
                inds = tf.placeholder('int64', shape=case[0],
                                      name='test_inds')
                val = unpool.unpool(x, inds, output_size=case[1])
                np.testing.assert_array_equal(
                    [a.value for a in val.shape], case[2])


if __name__ == '__main__':
    unittest.main()
