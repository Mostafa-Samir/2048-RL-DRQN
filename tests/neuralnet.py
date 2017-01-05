import unittest
import os
import tensorflow as tf
import numpy as np
from ai.nn.layer import *
from ai.nn.network import *

graph = tf.Graph()
session = tf.Session(graph=graph)

class NeuralNetTests(unittest.TestCase):

    def setUp(self):
        model_path = os.path.dirname(__file__) + "/test-model.ckpt"
        model_meta = os.path.dirname(__file__) + "/test-model.ckpt.meta"
        model_checkpoint = os.path.dirname(__file__) + "/checkpoint"
        if os.path.isfile(model_path):
            os.remove(model_path)
            os.remove(model_meta)
            os.remove(model_checkpoint)

    def tearDown(self):
        self.setUp()

    def test_construction(self):
        nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
            FCLayer(72, 10)
        ])

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

        nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
            FCLayer(72, 10)
        ])

        output = nn(sample_data)

        session.run(tf.initialize_all_variables())
        _output = session.run(output)

        self.assertEqual(_output.shape, (2, 10))

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

        nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
            FCLayer(72, 10)
        ])
        clone = nn.clone()

        output = nn(sample_data)
        clone_output = clone(sample_data)

        session.run(tf.initialize_all_variables())
        _output, _clone_output = session.run([output, clone_output])

        self.assertTrue(np.array_equal(_output, _clone_output))

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

        nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
            FCLayer(72, 10)
        ])
        other = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
            FCLayer(72, 10)
        ])

        output = nn(sample_data)
        before_assign_output = other(sample_data)

        session.run(tf.initialize_all_variables())
        _output, _before_assign_output = session.run([output, before_assign_output])

        other.assign_to(nn, session)
        after_assign_output = other(sample_data)

        _after_assign_output = session.run(after_assign_output)

        self.assertFalse(np.array_equal(_output, _before_assign_output))
        self.assertTrue(np.array_equal(_output, _after_assign_output))

    def test_save_restore(self):
        global session
        sample_data = tf.constant([[[[1.], [2.], [3.], [4.]],
                                   [[5.], [6.], [7.], [8.]],
                                   [[9.], [3.], [1.], [1.]],
                                   [[6.], [5.], [8.], [4.]]],
                                   [[[1.], [2.], [3.], [4.]],
                                    [[5.], [6.], [7.], [8.]],
                                    [[9.], [3.], [1.], [1.]],
                                    [[6.], [5.], [8.], [4.]]]])

        nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True, name='conv'),
            FCLayer(72, 10, name='fc')
        ])
        dummy_nn = NeuralNetwork([
            Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True, name='conv_dummy'),
            FCLayer(72, 10, name='fc_dummy')
        ])

        original_output = nn(sample_data)

        session.run(tf.initialize_all_variables())
        _original_output = session.run(original_output)

        model_path = os.path.dirname(__file__) + "/test-model.ckpt"
        nn.save(session, model_path)

        nn.assign_to(dummy_nn, session)
        corrupted_output = nn(sample_data)

        _corrupted_output = session.run(corrupted_output)

        nn.restore(session, model_path)

        restored_output = nn(sample_data)

        _restored_output = session.run(restored_output)

        self.assertFalse(np.array_equal(_original_output, _corrupted_output))
        self.assertTrue(np.array_equal(_original_output, _restored_output))



if __name__ == '__main__':
    with graph.as_default():
        with session:
            unittest.main()
