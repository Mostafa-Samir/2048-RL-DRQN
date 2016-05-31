import tensorflow as tf

from ai.dqn import DQN
from ai.dfnn import DFCNN

from utilis import expect_inrange
from os.path import dirname

graph = tf.Graph()

with graph.as_default():
    with tf.Session(graph=graph) as session:

        qnn = DFCNN([9, 5, 4])
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        summary_writer = tf.train.SummaryWriter(dirname(__file__) + "/../tflogs")
        trainer = DQN(qnn, optimizer, session, 9, 4, summary_writer=summary_writer, minibatch_size=1)

        tf.initialize_all_variables().run()

        print "# Initialization with no Errors: Passed!"

        state = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        action = 2
        reward = 4
        nextstate = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        legals = [0,2]

        a = trainer.get_action(state, [0,3])
        print '# Correct Action Retrival Behavior (Train Mode):'
        expect_inrange(a, [0,3])

        trainer.remember(state, action, reward, nextstate, legals)
        print "# Store Experience with no Errors: Passed!"
        trainer.train()
        print "# Train with no Errors: Passed!"

        a = trainer.get_action(state, [0,1,3], True)
        print '# Correct Action Retrival Behavior (Play Mode):'
        expect_inrange(a, [0,1,3])
