from Layer import Layer


class NeuralNetwork:
    def __init__(self, weights, biases, activation_functions):
        layer_count = len(weights)

        if (len(biases) != layer_count
                or len(activation_functions) != layer_count):
            raise ValueError

        self.layers = []

        for i in range(layer_count):
            layer = Layer(weights[i], biases[i], activation_functions[i])
            self.layers.append(layer)

    def propagate(self, nn_inputs):
        inputs = nn_inputs

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def getLayers(self):
        return self.layers
