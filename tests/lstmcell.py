import unittest
import tensorflow as tf
import numpy as np
from ai.nn.layer import LSTMCell

graph = tf.Graph()
session = tf.Session(graph=graph)

class LSTMCellTests(unittest.TestCase):

    def test_construction(self):
        cell = LSTMCell(16, 4, 2)

        self.assert_(True)

    def test_evaluation(self):
        global session
        sample_data = tf.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])

        cell = LSTMCell(4, 8, 2)

        session.run(tf.initialize_all_variables())

        cell(sample_data)
        self.assert_(True)

    def test_assignment(self):
        global session
        sample_data = tf.constant([[10., 20., 30., 40.], [50., 60., 70., 80.]])

        cell = LSTMCell(4, 8, 2)
        other = LSTMCell(4, 8, 2)

        session.run(tf.initialize_all_variables())

        cell_output, other_output = session.run([cell(sample_data), other(sample_data)])

        other.assign_to(cell, session)

        after_assign_cell_output, after_assign_other_output = session.run([cell(sample_data), other(sample_data)])

        self.assertFalse(np.array_equal(cell_output, other_output))
        self.assertTrue(np.array_equal(after_assign_cell_output, after_assign_other_output))

    def test_cloning(self):
        global session
        sample_data = tf.constant([[10., 20., 30., 40.], [50., 60., 70., 80.]])

        cell = LSTMCell(4, 8, 2)
        clone = cell.clone()

        session.run(tf.initialize_all_variables())

        cell_output, clone_output = session.run([cell(sample_data), clone(sample_data)])

        self.assertTrue(np.array_equal(cell_output, clone_output))

if __name__ == '__main__':
    with graph.as_default():
        with session:
            unittest.main()
