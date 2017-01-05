import unittest
import tensorflow as tf
import numpy as np
from ai.nn.layer import FCLayer

graph = tf.Graph()
session = tf.Session(graph=graph)

class FCLayerTests(unittest.TestCase):

    def test_construction(self):
        layer = FCLayer(5, 3)

        self.assert_(True)

    def test_evaluation(self):
        global session
        sample_data = tf.constant([[6., 7., 8., 9., 10.], [1., 2., 3., 4., 5.]])

        layer = FCLayer(5, 3)
        output = layer(sample_data)

        session.run(tf.initialize_all_variables())
        _output = session.run([output])

        self.assert_(True)

    def test_cloning(self):
        sample_data = tf.constant([[6., 7., 8., 9., 10.], [1., 2., 3., 4., 5.]])

        layer = FCLayer(5, 3)
        layer_output = layer(sample_data)
        clone = layer.clone()
        clone_output = clone(sample_data)

        session.run(tf.initialize_all_variables())
        _layer_output, _clone_output = session.run([layer_output, clone_output])

        self.assertTrue(np.array_equal(_layer_output, _clone_output))

    def test_assignment(self):
        global session
        sample_data = tf.constant([[6., 7., 8., 9., 10.], [1., 2., 3., 4., 5.]])

        layer = FCLayer(5, 3)
        layer_output = layer(sample_data)
        other = FCLayer(5, 3)
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
