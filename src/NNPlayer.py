import numpy

from NeuralNetwork import NeuralNetwork


GRID_COUNT = 3
ROW_LENGTH = 3
COLUMN_LENGTH = 3

INPUT_SIZE = GRID_COUNT * 2 + ROW_LENGTH * 2 + COLUMN_LENGTH * 2
OUTPUT_SIZE = GRID_COUNT * ROW_LENGTH * COLUMN_LENGTH


class NNPlayer:
    @staticmethod
    def getSpecs():
        return (INPUT_SIZE, OUTPUT_SIZE)

    def __init__(self, weights, biases, activation_functions):
        self.neural_network = NeuralNetwork(
            weights,
            biases,
            activation_functions
        )

    def play(self, player_state, opponent_state,
             player_score, opponent_score,
             turn, length, pip_count):
        player_state = numpy.array(player_state)
        opponent_state = numpy.array(opponent_state)

        # Sum each column, row, and grid
        column_sums = numpy.ravel(
            [player_state.sum((0, 1)), -(opponent_state.sum((0, 1)))], "F"
        )
        row_sums = numpy.ravel(
            [player_state.sum((0, 2)), -(opponent_state.sum((0, 2)))], "F"
        )
        grid_sums = numpy.ravel(
            [player_state.sum((1, 2)), -(opponent_state.sum((1, 2)))], "F"
        )

        nn_inputs = numpy.concatenate((column_sums, row_sums, grid_sums))

        # Run neural network for current turn
        output = self.neural_network.propagate(nn_inputs)

        # Index on board as if a 1D array
        board_position = int(numpy.argmax(output))

        # Convert 1D index to grid, row, and column indexes
        grid = board_position % GRID_COUNT
        row = (board_position // GRID_COUNT) % ROW_LENGTH
        column = board_position // (ROW_LENGTH * COLUMN_LENGTH)

        return [grid, row, column]

    def getNN(self):
        return self.neural_network
