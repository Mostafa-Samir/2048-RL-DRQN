import unittest
import os
import tensorflow as tf
import numpy as np
from ai.nn.layer import *
from ai.nn.network import *
from ai.dqn import *

graph = tf.Graph()
session = tf.Session(graph=graph)

class DQNTests(unittest.TestCase):

    def test_construction(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                qnn = NeuralNetwork([
                    Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
                    FCLayer(72, 4)
                ])

                optimizer = tf.train.GradientDescentOptimizer(0.1)
                summary_writer = tf.train.SummaryWriter(os.path.dirname(__file__) + "/../tflogs")
                trainer = DQN(qnn, optimizer, session, [4, 4, 1], 4, summary_writer=summary_writer, minibatch_size=1)

                self.assert_(True)

    def test_action(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                qnn = NeuralNetwork([
                    Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
                    FCLayer(8, 4)
                ])

                optimizer = tf.train.GradientDescentOptimizer(0.1)
                summary_writer = tf.train.SummaryWriter(os.path.dirname(__file__) + "/../tflogs")
                trainer = DQN(qnn, optimizer, session, [2, 2, 1], 4, summary_writer=summary_writer, minibatch_size=1)

                state = [[1., 2.], [3., 4.]]
                legals = [0, 1]

                session.run(tf.initialize_all_variables())

                action = trainer.get_action(state, legals)

                self.assertTrue(action in legals)

    def test_remember(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                qnn = NeuralNetwork([
                    Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
                    FCLayer(8, 4)
                ])

                optimizer = tf.train.GradientDescentOptimizer(0.1)
                summary_writer = tf.train.SummaryWriter(os.path.dirname(__file__) + "/../tflogs")
                trainer = DQN(qnn, optimizer, session, [2, 2, 1], 4, summary_writer=summary_writer, minibatch_size=1)

                state = [[1., 2.], [3., 4.]]
                action = 2
                reward = 1.5
                nextstate = [[3., 4.], [5., 6.]]
                legals = [0, 1]

                trainer.remember(state, action, reward, nextstate, legals)

                self.assertEqual(trainer.experience[0][0].shape, (2, 2, 1))

    def test_train(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as session:

                qnn = NeuralNetwork([
                    Conv2DLayer([2, 2], 1, 8, [1, 1], padding='VALID', flatten=True),
                    FCLayer(8, 4)
                ])

                optimizer = tf.train.GradientDescentOptimizer(0.1)
                summary_writer = tf.train.SummaryWriter(os.path.dirname(__file__) + "/../tflogs")
                trainer = DQN(qnn, optimizer, session, [2, 2, 1], 4, summary_writer=summary_writer, minibatch_size=1)

                state = [[1., 2.], [3., 4.]]
                action = 2
                reward = 1.5
                nextstate = [[3., 4.], [5., 6.]]
                legals = [0, 1]

                tf.initialize_all_variables().run()

                trainer.remember(state, action, reward, nextstate, legals)
                trainer.train()

                self.assert_(True)


if __name__ == '__main__':
    unittest.main()
