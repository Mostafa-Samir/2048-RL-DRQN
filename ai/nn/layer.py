import tensorflow as tf
import numpy as np
from math import sqrt

activations_dict = {
    'relu': tf.nn.relu,
    'tanh': tf.tanh,
    'sigmoid': tf.sigmoid
}

class FCLayer:

    # a static variable to keep count of FCLayers created
    created_count = 0

    def __init__(self, input_size, output_size, activation='relu', name='fclayer'):
        """
        constructs a fully-connected layer

        Parameters:
        ----------
        input_size: int
            the size of the input vector to the layer
        output_size: int
            the size of the output vector from the layer
        activation: string
            the name of the activation function
        name: string
            the name of the layer (useful for saving and loading a model)
        """

        global activations_dict

        FCLayer.created_count += 1

        self.id = name + '_' + str(FCLayer.created_count) if name == 'fclayer' else name
        self.input_size = input_size
        self.output_size = output_size
        self.stddev = min(0.02, sqrt(2. / input_size))
        self.activation = activation
        self.activation_fn = activations_dict[activation]

        self.weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=self.stddev), name=self.id + "_w")
        self.bias = tf.Variable(tf.zeros([output_size]), name=self.id + "_b")

    def __call__(self, X):
        """
        computes and returns W^TX + b when the object is called

        Parameters:
        ----------
        X: Tensor
            the input vector to compute the layer output on

        Returns: Tensor
            W^TX+ b
        """

        return self.activation_fn(tf.matmul(X, self.weights) + self.bias)

    def assign_to(self, other, session):
        """
        Assigns the parameters of the current layer to an other

        Parameters:
        ----------
        other: FCLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """
        if isinstance(other, FCLayer) and self.input_size == other.input_size and self.output_size == other.output_size:
            weights_assign = self.weights.assign(other.weights)
            bias_assign = self.bias.assign(other.bias)

            session.run([weights_assign, bias_assign])
        else:
            raise TypeError("Cannot assign FCLayer: mismatch in type or size")


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """

        clone = FCLayer(self.input_size, self.output_size, self.activation, self.id + '_clone')
        clone.weights = tf.Variable(self.weights.initialized_value(), name=self.id + '_clone_w')
        clone.bias = tf.Variable(self.bias.initialized_value(), name=self.id + '_clone_b')

        return clone


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """

        return [self.weights, self.bias]


class Conv2DLayer:

    # a static variable to keep count of Conv2DLayer created
    created_count = 0

    def __init__(self, patch, in_chs, out_chs, strides, padding, activation='relu', name='convlayer'):
        """
        Constructs a convolutional layer

        Parameters:
        ----------
        patch: list/tuple [2 elements]
            The dimensions of the convolutional filter
        in_chs: int
            The number of channels in input data
        out_chs: int
            The number of channels in output data
        strides: list/tuple [2 elements]
            The amount of units slided by the filter in each direction
        padding: string ['VALID', 'SAME']
            The type padding used in the convolution
        activation: string
            The name of the activation function
        name: string
            The name of the layer (useful for saving and loading models)
        """

        global activations_dict

        Conv2DLayer.created_count += 1

        self.id = name + '_' + str(Conv2DLayer.created_count) if name == 'convlayer' else name
        self.patch_dims = patch
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.activation_fn = activations_dict[activation]

        weights_shape = [self.patch_dims[0], self.patch_dims[1], self.in_channels, self.out_channels]
        self.weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.05), name=self.id + '_w')
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.out_channels]), name=self.id + '_b')

    def __call__(self, X):
        """
        Applys the layer's convolution to input X along with activation
        and output reshaping if specified

        Parameters:
        ----------
        X: Tensor
            The input data to the layer

        Returns: Tensor
            The output from the convolution operation
        """
        _strides = [1, self.strides[0], self.strides[1], 1]
        convolved = tf.nn.conv2d(X, self.weights, _strides, padding=self.padding)
        convolved = convolved + self.bias
        activations = self.activation_fn(convolved)

        return activations

    def assign_to(self, other, session):
        """
        Assigns the parameters of the current layer to an other

        Parameters:
        ----------
        other: Conv2DLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """
        convolution_match = set(self.patch_dims) == set(other.patch_dims)
        size_match = self.in_channels == other.in_channels and self.out_channels == other.out_channels
        if isinstance(other, Conv2DLayer) and convolution_match and size_match:
            weights_assign = self.weights.assign(other.weights)
            bias_assign = self.bias.assign(other.bias)

            session.run([weights_assign, bias_assign])
        else:
            raise TypeError("Cannot Assign Conv2DLayer: type or size mismatch")

    def clone(self):
        """
        Clones the current layer into a new layer instance
        """

        clone = Conv2DLayer(self.patch_dims, self.in_channels,
                            self.out_channels, self.strides, self.padding,
                            self.activation, self.id + '_clone')
        clone.weights = tf.Variable(self.weights.initialized_value(), name=self.id + '_clone_w')
        clone.bias = tf.Variable(self.bias.initialized_value(), name=self.id + '_clone_b')

        return clone


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """

        return [self.weights, self.bias]

class LSTMCell:

    # a static variable to keep count of LSTMCell created
    created_count = 0

    def __init__(self, input_size, num_hidden, minibatch_size, name='lstmcell'):
        """
        Constructs an LSTM Cell

        Parameters:
        ----------
        input_size: int
            the size of the single input vector to the cell
        num_hidden: int
            the number of hidden nodes in the cell
        minibatch_size: int
            the number of the input vectors in the input matrix
        """

        LSTMCell.created_count += 1

        self.id = name + '_' + str(LSTMCell.created_count) if name == 'lstmcell' else name

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.minibatch_size = minibatch_size

        self.input_weights = tf.Variable(tf.truncated_normal([self.input_size, self.num_hidden * 4], -0.1, 0.1), name=self.id + '_wi')
        self.output_weights = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_hidden * 4], -0.1, 0.1), name=self.id + '_wo')
        self.bias = tf.Variable(tf.zeros([self.num_hidden * 4]), name=self.id + '_b')

        self.prev_output = tf.Variable(tf.zeros([self.minibatch_size, self.num_hidden]), trainable=False, name=self.id+'_o')
        self.prev_state = tf.Variable(tf.zeros([self.minibatch_size, self.num_hidden]), trainable=False, name=self.id+'_s')

    def __call__(self, X):
        """
        Performs the LSTM's forget, input and output operations
        according to: http://arxiv.org/pdf/1402.1128v1.pdf without peepholes

        Parameters:
        ----------
        X: list[Tensor]
            The input list to process by the LSTM
        """
        outputs = tf.TensorArray(tf.float32, len(X))
        inputs = tf.TensorArray(tf.float32, len(X))
        t = tf.constant(0, dtype=tf.int32)

        for i, step_input in enumerate(X):
            inputs = inputs.write(i, step_input)

        def step_op(time, prev_state, prev_output, inputs_list, outputs_list):
            time_step = inputs_list.read(time)
            gates = tf.matmul(time_step, self.input_weights) + tf.matmul(prev_output, self.output_weights) + self.bias
            gates = tf.reshape(gates, [-1, self.num_hidden, 4])

            input_gate = tf.sigmoid(gates[:, :, 0])
            forget_gate = tf.sigmoid(gates[:, :, 1])
            candidate_state = tf.tanh(gates[:, :, 2])
            output_gate = tf.sigmoid(gates[:, :, 3])

            state = forget_gate * prev_state + input_gate * candidate_state
            output = output_gate * tf.tanh(state)
            new_outputs = outputs_list.write(time, output)

            return time + 1, state, output, inputs_list, new_outputs

        _, state, output, _, final_outputs = tf.while_loop(
            cond=lambda time, *_: time < len(X),
            body= step_op,
            loop_vars=(t, self.prev_state, self.prev_output, inputs, outputs),
            parallel_iterations=32,
            swap_memory=True
        )

        self.prev_state.assign(state)
        self.prev_output.assign(output)

        return [final_outputs.read(t) for t in range(len(X))]

    def assign_to(self, other, session):
        """
        Assigns the parameters of the cuurrent cell to another's

        Parameters:
        ----------
        other: LSTMCell
            The cell to darw the parameters from
        session: tf.Session
            The tensorflow session to run the assignments
        """
        shape_set = set([self.input_size, self.num_hidden, self.minibatch_size])
        other_shape_set = set([other.input_size, other.num_hidden, other.minibatch_size])

        if isinstance(other, LSTMCell) and shape_set == other_shape_set:
            input_weights_assign = self.input_weights.assign(other.input_weights)
            output_weights_assign = self.output_weights.assign(other.output_weights)
            bias_assign = self.bias.assign(other.bias)
            prev_state_assign = self.prev_state.assign(other.prev_state)
            prev_output_assign = self.prev_output.assign(other.prev_output)

            session.run([input_weights_assign, output_weights_assign, bias_assign, prev_state_assign, prev_output_assign])
        else:
            raise TypeError("Cannot assign an LSTMCell: type or size mismatch")

    def clone(self):
        """
        Clones the current cell to another LSTMCell instance
        """

        clone = LSTMCell(self.input_size, self.num_hidden, self.minibatch_size, self.id + '_clone')
        clone.input_weights = tf.Variable(self.input_weights.initialized_value(), name=self.id + '_clone_wi')
        clone.output_weights = tf.Variable(self.output_weights.initialized_value(), name=self.id + '_clone_wo')
        clone.bias = tf.Variable(self.bias.initialized_value(), name=self.id + '_clone_b')
        clone.prev_state = tf.Variable(self.prev_state.initialized_value(), trainable=False, name=self.id + '_clone_s')
        clone.prev_output = tf.Variable(self.prev_output.initialized_value(), trainable=False, name=self.id + '_clone_o')

        return clone

    def clear(self, session):
        """
        clears the hidden state of the LSTM
        """
        zero_state = self.prev_state.assign(np.zeros((self.minibatch_size, self.num_hidden), dtype=np.float32))
        zero_output = self.prev_output.assign(np.zeros((self.minibatch_size, self.num_hidden), dtype=np.float32))

        session.run([zero_state, zero_output])

    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """

        return [
            self.input_weights,
            self.output_weights,
            self.bias,
            self.prev_state,
            self.prev_output
        ]
