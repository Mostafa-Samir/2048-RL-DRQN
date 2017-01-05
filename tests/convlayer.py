import unittest
import tensorflow as tf
import numpy as np
from ai.nn.layer import Conv2DLayer

graph = tf.Graph()
session = tf.Session(graph=graph)

class ConvLayerTests(unittest.TestCase):

    def test_construction(self):
        layer = Conv2DLayer([2, 2], 1, 8, [2, 2], padding='SAME')

        self.assert_(True)

    def test_evaluation(self):
        global session
        sample_data = tf.constant([[[[1.], [2.], [3.], [4.]],
                                   [[5.], [6.], [7.], [8.]],
                                   [[9.], [3.], [1.], [1.]],
                                   [[6.], [5.], [8.], [4.]]],
                                   [[[1.], [2.], [3.], [4.]],
                                    [[5.], [6.], [7.], [8.]],
                                    [[9.], [3.], [1.], [1.]],
                                    [[6.], [5.], [8.], [4.]]]])

        slayer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='SAME')
        vlayer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID')
        sflayer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='SAME', flatten=True)
        vflayer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True)

        sout = slayer(sample_data)
        vout = vlayer(sample_data)
        sfout = sflayer(sample_data)
        vfout = vflayer(sample_data)

        session.run(tf.initialize_all_variables())
        _sout, _vout, _sfout, _vfout = session.run([sout, vout, sfout, vfout])

        self.assertEqual(_sout.shape, (2, 4, 4, 8))
        self.assertEqual(_vout.shape, (2, 3, 3, 8))
        self.assertEqual(_sfout.shape, (2, 128))
        self.assertEqual(_vfout.shape, (2, 72))

    def test_cloning(self):
        global session
        sample_data = tf.constant([[[[1.], [2.], [3.], [4.]],
                                   [[5.], [6.], [7.], [8.]],
                                   [[9.], [3.], [1.], [1.]],
                                   [[6.], [5.], [8.], [4.]]],
                                   [[[1.], [2.], [3.], [4.]],
                                    [[5.], [6.], [7.], [8.]],
                                    [[9.], [3.], [1.], [1.]],
                                    [[6.], [5.], [8.], [4.]]]])

        layer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID')
        clone = layer.clone()

        layer_output = layer(sample_data)
        clone_output = clone(sample_data)

        session.run(tf.initialize_all_variables())
        _layer_output, _clone_output = session.run([layer_output, clone_output])

        self.assertTrue(np.array_equal(_layer_output, _clone_output))

    def test_assignment(self):
        global session
        sample_data = tf.constant([[[[1.], [2.], [3.], [4.]],
                                   [[5.], [6.], [7.], [8.]],
                                   [[9.], [3.], [1.], [1.]],
                                   [[6.], [5.], [8.], [4.]]],
                                   [[[1.], [2.], [3.], [4.]],
                                    [[5.], [6.], [7.], [8.]],
                                    [[9.], [3.], [1.], [1.]],
                                    [[6.], [5.], [8.], [4.]]]])

        layer = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID')
        other = Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID')

        layer_output = layer(sample_data)
        before_assign_output = other(sample_data)

        session.run(tf.initialize_all_variables())
        _layer_output, _before_assign_output = session.run([layer_output, before_assign_output])

        other.assign_to(layer, session)
        after_assign_output = other(sample_data)

        _after_assign_output = session.run(after_assign_output)

        self.assertFalse(np.array_equal(_layer_output, _before_assign_output))
        self.assertTrue(np.array_equal(_layer_output, _after_assign_output))

if __name__ == '__main__':
    with graph.as_default():
        with session:
            unittest.main()
