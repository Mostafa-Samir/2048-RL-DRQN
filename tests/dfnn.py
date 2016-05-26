from ai.dfnn import Layer, DFCNN
import tensorflow as tf

import os

from utilis import expect_equal, expect_not_equal

graph = tf.Graph()

with graph.as_default():

    sample_data = tf.constant([[6., 7., 8., 9., 10.], [1., 2., 3., 4., 5.]])
    layer_nn = Layer(5, 3, 5)

    layer = Layer(5, 1, indx=-1)
    layer_clone = layer.clone()

    value = layer(sample_data)
    cvalue = layer_clone(sample_data)

    nn = DFCNN([5, 3, 1])
    nn_clone = nn.clone()

    nnvalue = nn(sample_data)
    nncvalue = nn_clone(sample_data)

with tf.Session(graph=graph) as session:
    session.run(tf.initialize_all_variables())
    _value,_cvalue, _nnvalue, _nncvalue = session.run([value, cvalue, nnvalue, nncvalue])

    print "#Layer: Cloning At Initialization: "
    expect_equal(_value, _cvalue)

    print "#NN: Cloning At Initialization:"
    expect_equal(_nnvalue, _nncvalue)

    # changing the weigths of a layer after initialization
    assigner = layer.weights.assign(layer.weights * 2)
    session.run([assigner])

    layer_clone.assign_to(layer, session)

    value2 = layer(sample_data)
    cvalue2 = layer_clone(sample_data)

    nn.save(session, os.path.dirname(__file__) + "/../tflogs/save-test.ckpt")

    # changing a layer in the neural network after initialization
    nn.layers[0].assign_to(layer_nn, session)

    nn_clone.assign_to(nn, session)

    nnvalue2 = nn(sample_data)
    nncvalue2 = nn_clone(sample_data)

    _value2, _cvalue2, _nnvalue2, _nncvalue2 = session.run([value2, cvalue2, nnvalue2, nncvalue2])

    print "#Layer: Assignment After Initialization: "
    expect_equal(_value2, _cvalue2)

    print "#Layer: Assigned Layer is Different than original:"
    expect_not_equal(_value2, _value)

    print "#NN: Cloning After Initialization:"
    expect_equal(_nnvalue2, _nncvalue2)
    print "#NN: Assigned NN is Different than original:"
    expect_not_equal(_nnvalue2, _nnvalue)

    nn.restore(session, os.path.dirname(__file__) + "/../tflogs/save-test.ckpt")

    nnrvalue = nn(sample_data)
    _nnrvalue = session.run(nnrvalue)

    print "#NN: Saving and Restoring:"
    expect_equal(_nnvalue, _nnrvalue)
