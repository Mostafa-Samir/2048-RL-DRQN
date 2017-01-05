from ai.nn.layer import *
from ai.nn.helpers import *
from ai.nn.network import *
from ai.dqn import DQN
from ai.drqn import DRQN

def drqn_model1(session):
    batch_size = 16
    time_steps = 16

    qnn = NeuralNetwork([
        Reshape(lambda (A,B,C,D,E): (batch_size*time_steps,C,D,E)),
        Merge([
            Sequence([
                Conv2DLayer([4, 1], 16, 32, [4, 1], padding='VALID'),
                Reshape(lambda (AB,C,D,E): (AB, C*D*E))
            ]),
            Sequence([
                Conv2DLayer([1, 4], 16, 32, [1, 4], padding='VALID'),
                Reshape(lambda (AB,C,D,E): (AB, C*D*E))
            ]),
            Sequence([
                Conv2DLayer([2, 2], 16, 32, [1, 1], padding='VALID'),
                Reshape(lambda (AB,C,D,E): (AB, C*D*E))
            ])
        ], axis = 1),
        Reshape(lambda (AB, CDE): (batch_size, time_steps, CDE)),
        Unroll(axis=1, num=time_steps),
        LSTMCell(544, 256, minibatch_size=batch_size),
        Roll(axis=1),
        Reshape(lambda (A, B, CDE): (batch_size * time_steps, CDE)),
        FCLayer(256, 4)
    ])

    optimizer = tf.train.AdamOptimizer(0.001)

    trainer = DRQN(qnn, optimizer, session, [4, 4, 16], 4,
        final_exploration_probability=0.05,
        exploration_period=1000,
        reply_memory_size=48,
        target_freeze_period= 2500,
        unrollings_num=time_steps,
        minibatch_size=batch_size)

    return qnn, trainer, '4x4x16', 'log2Plain'


def drqn_model2(session):

    qnn, trainier, state_representation, _ = drqn_model1(session)
    return qnn, trainier, state_representation, 'log2MaxTileEmptyDiff'
