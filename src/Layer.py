import numpy


def tanh(a):
    return numpy.tanh(a)


def linear(a):
    return a


def relu(a):
    return numpy.maximum(0, a)


class Layer():
    def __init__(self, weights, biases, activation_function):
        if len(biases) != len(weights):
            raise ValueError

        self.weights = numpy.copy(weights)
        self.biases = numpy.copy(biases)
        self.activate = activation_function

    def forward(self, inputs):
        return self.activate(numpy.dot(self.weights, inputs) + self.biases)

    def getMatrix(self):
        return self.weights

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        return self.activate
