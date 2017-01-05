import tensorflow as tf
from layer import LSTMCell

class NeuralNetwork:

    def __init__(self, architecture):
        """
        Constructs a neural network

        Parameters:
        -----------
        architecture: list
            The list of layers in the network
        """

        self.layers = architecture

    def __call__(self, X):
        """
        Evaluates and returns the neural network on the input X

        Parameters:
        ----------
        X: Tensor
            The input data to evaluate the network on

        Returns: Tensor
            The evaluation result
        """
        output = X
        for layer in self.layers:
            output = layer(output)

        return output

    def clone(self):
        """
        Clones the current network into a new network instance
        """

        clone = NeuralNetwork([])

        for layer in self.layers:
            clone.layers.append(layer.clone())

        return clone

    def assign_to(self, other, session):
        """
        Assigns the current network's parameters to another network's parameters'

        Parameters:
        -----------
        other: NeuralNetwork
            the network to draw the parameters from
        session: tf.Session
            the tensorflow session to run the assignments
        """

        if len(self.layers) == len(other.layers):
            for i in range(len(self.layers)):
                self.layers[i].assign_to(other.layers[i], session)
        else:
            print "Network Mismatch"

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
            var_list.extend(layer.get_variables())

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
            var_list.extend(layer.get_variables())

        saver = tf.train.Saver(var_list)
        saver.restore(session, load_path)

    def clearLSTMS(self, session):
        """
        clears any lstm cell in the network
        """
        for layer in self.layers:
            if isinstance(layer, LSTMCell):
                layer.clear(session)
