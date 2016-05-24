import tensorflow as tf
from math import sqrt

class Layer:

    def __init__(self, input_size, output_size):
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
        self.input_size = input_size
        self.output_size = output_size

        self.weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=sqrt(2. / input_size)))
        self.bias = tf.Variable(tf.zeros([output_size]))


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

        layer_clone = Layer(self.input_size, self.output_size)
        layer_clone.weights = layer_clone.weights.assign(self.weights)
        layer_clone.bias = layer_clone.bias.assign(self.bias)

        return layer_clone


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
                self.layers.append(Layer(input_size, output_size))


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


    def clone(self):
        """
        returns a deep copy of the network
        """

        nn_clone = DFCNN([], self.activation_fn_name)

        for layer in self.layers:
            nn_clone.layers.append(layer.clone())

        return nn_clone
