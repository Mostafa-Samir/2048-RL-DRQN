import tensorflow as tf
from math import sqrt

class Layer:

    def __init__(self, input_size=0, output_size=0, indx = 0, copy_from=None):
        """
        constructs a layer in a deep fully-connected neural network

        Parameters:
        ----------
        input_size: int
            the size of the input to the layer
        output_size: int
            the size of the output from the layer
        indx: int
            the index of the layer in the neural network
        """
        if copy_from is None:
            self.name = "layer" + str(indx)
            self.input_size = input_size
            self.output_size = output_size

            self.weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=sqrt(2. / input_size)), name=self.name + "_weights")
            self.bias = tf.Variable(tf.zeros([output_size]), name=self.name + "_bias")
        else:
            source = copy_from

            self.name = source.name + "-copy"
            self.input_size = source.input_size
            self.output_size = source.output_size

            self.weights = tf.Variable(source.weights.initialized_value(), name=self.name + "_weights")
            self.bias = tf.Variable(source.bias.initialized_value(), name=self.name + "_bias")


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
        return tf.matmul(X, self.weights) + self.bias


    def clone(self):
        """
        returns a deep copy of the layer
        """

        layer_clone = Layer(copy_from=self)

        return layer_clone

    def assign_to(self, other, session):
        """
        assigns the weigts and the biases of the layer to the
        values of another

        Parameters:
        ----------
        other: Layer
            The target layer
        session: tf.Session
            a session to run the assignments
        """

        if(self.input_size == other.input_size and self.output_size == other.output_size):
            weights_assign = self.weights.assign(other.weights)
            bias_assign = self.bias.assign(other.bias)

            session.run([weights_assign, bias_assign])
        else:
            print "Layer Mismatch!"


class DFCNN:

    def __init__(self, architecture, activation='relu'):
        """
        constructs a deep fully-connected neural network

        Parameters:
        -----------
        architecture: list
            a list of the number of hidden nodes in each layer
            the first element is the input size, while the last
            is the output_size
        activation: string
            the name of the activation function to be used
            possible values: sigmoid, tanh, relu
            defualt value: 'relu'
        """

        available_activations = {
            'relu': tf.nn.relu,
            'tanh': tf.tanh,
            'sigmoid': tf.sigmoid
        }

        self.activation_fn_name = activation
        self.activation_fn = available_activations[activation]
        self.layers = [];

        if(len(architecture) > 0):
            for i, hidden_nodes in enumerate(architecture[: len(architecture) - 1]):
                input_size, output_size = architecture[i], architecture[i + 1]
                self.layers.append(Layer(input_size, output_size, i))


    def __call__(self, X):
        """
        Performs a forward run of the neural network and retunrns the output

        Parameters:
        ----------
        X: Tensor
            the input data to feed forward the neural network
        Returns: Tensor
            A tensor of the output size shape
        """
        hidden_layers, output_layer = self.layers[:len(self.layers) - 1], self.layers[len(self.layers) - 1]
        activations = X
        for layer in hidden_layers:
            activations = self.activation_fn(layer(activations))

        return output_layer(activations)

    def assign_to(self, other, session):
        """
        assigns the weigths and biases of the layers to the values of
        another network

        Parameters:
        ----------
        other: DFCNN
            the target network
        session: tf.Session
            a session to run the assignments
        """
        if(len(self.layers) == len(other.layers)):
            for i in range(len(self.layers)):
                self.layers[i].assign_to(other.layers[i], session)
        else:
            print "Network Mismatch!"

    def clone(self):
        """
        returns a deep copy of the network
        """

        nn_clone = DFCNN([], self.activation_fn_name)

        for layer in self.layers:
            nn_clone.layers.append(layer.clone())

        return nn_clone

    def save(self, session, save_path):
        """
        saves the neural network to disk

        Parameters:
        ----------
        session: tf.Session
            The tensor flow session enclosing the operations
        save_path: string
            The path to save the neural network in
        """

        var_list = []

        for layer in self.layers:
            var_list.extend([layer.weights, layer.bias])

        saver = tf.train.Saver(var_list)
        saver.save(session, save_path)

    def restore(self, session, load_path):
        """
        restores a saved neural network from disk

        Parameters:
        ----------
        session: tf.Session
            The tensor flow session enclosing the operations
        load_path: string
            The path to load the neural network from
        """

        var_list = []

        for layer in self.layers:
            var_list.extend([layer.weights, layer.bias])

        saver = tf.train.Saver(var_list)
        saver.restore(session, load_path);
