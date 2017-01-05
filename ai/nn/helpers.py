import tensorflow as tf

class _CompositeLayer:


    def __init__(self, components, op):

        """
        constructs a layer that operates as a composite of other primitive layers

        Parameters:
        ----------
        components: list
            the list of layers participating in the composition
        op: function
            the composite operation of the layer
        """

        self.components = components
        self.op = op


    def __call__(self, X):
        """
        computes and returns the the output of the layer

        Parameters:
        ----------
        X: Tensor
            The input tensor

        Returns: Tensor
            defined by self.op
        """

        return self.op(self, X)

    def assign_to(self, other, session):
        """
        Assigns the current layer to an other

        Parameters:
        ----------
        other: _CompositeLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """

        if isinstance(other, _CompositeLayer) and other.op == self.op:
            for i, component in enumerate(self.components):
                component.assign_to(other.components[i], session)
        else:
            raise TypeError("Cannot assign _CompositeLayer: mismatch in type or op")


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """
        cloned_components = [component.clone() for component in self.components]
        clone = _CompositeLayer(cloned_components, self.op)

        return clone


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """
        var_list = []

        for layer in self.components:
            var_list.extend(layer.get_variables())

        return var_list

class _OperationalLayer:


    def __init__(self, op, params):
        """
        constrcuts a layer that merely transform data as the propgagte
        through the network

        Parameters:
        ----------
        op: function
            The operation to be done on the data
        params: list
            the list of parameters needed to perform the op
        """
        self.op = op
        self.params = params


    def __call__(self, X):
        """
        Performs the defined operation on the input data

        Parameters:
        ----------
        X: Tensor
        Returns: Tensor
        """
        return self.op(self, X)


    def assign_to(self, other, session):
        """
        Assigns the current layer to an other

        Parameters:
        ----------
        other: _CompositeLayer
            the other layer to be assigned
        session: tf.Session
            the tensorflow session that will run the assignment
        """
        if not isinstance(other, _OperationalLayer) or not self.op == other.op:
            raise TypeError("Cannot assign _OperationalLayer: mismatch in type or op")
        else:
            self.params = other.params


    def clone(self):
        """
        Clones the current layer into a new layer instance
        """
        return _OperationalLayer(self.op, self.params[:])


    def get_variables(self):
        """
        gets the variables of the layer

        Returns: list
        """
        return []


def Sequence(layers):
    """
    defines a _CompositeLayer that runs component layers in sequence

    Parameters:
    ----------
    layers: list
        component layers
    """

    def sequence_op(obj, X):
        output = X
        for layer in obj.components:
            output = layer(output)

        return output

    return _CompositeLayer(layers, sequence_op)


def Merge(layers, axis):
    """
    defines a _CompositeLayer that merges the output of component layers

    Parameters:
    ----------
    layers: list
        component layers
    axis: int
        the axis to merge on
    """

    def merge_op(obj, X):
        return tf.concat(axis, [layer(X) for layer in obj.components])

    return _CompositeLayer(layers, merge_op)


def Reshape(new_shape):
    """
    defines an _OperationalLayer that reshapes the input to new_shape

    Parameters:
    ----------
    new_shape: list | tuple | function
    Returns: _OperationalLayer
    """

    def reshape_op(obj, X):
        dummy_lambda = lambda x:x
        new_shape = obj.params[0]
        if isinstance(obj.params[0], type(dummy_lambda)):
            old_shape = X.get_shape().as_list()
            new_shape = obj.params[0](old_shape)

        return tf.reshape(X, new_shape)

    return _OperationalLayer(reshape_op, [new_shape])


def Unroll(axis, num=None):
    """
    defines an _OperationalLayer that unpacks a tensor along a given axis

    Parameters:
    ----------
    axis: int
    num: int
        the numeber if tensors to unpack form the gievn tensor
    Returns: _OperationalLayer
    """

    def unroll_op(obj, X):
        return tf.unpack(X, obj.params[0], 1)

    return _OperationalLayer(unroll_op, [num, axis])


def Roll(axis):
    """
    defines an _OperationalLayer that packs a list of tensors on a given axis

    Parameters:
    -----------
    axis: int
    Returns: _OperationalLayer
    """

    def roll_op(obj, X):
        return tf.pack(X, axis=obj.params[0])

    return _OperationalLayer(roll_op, [axis])
