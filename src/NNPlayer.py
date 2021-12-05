import numpy

from NeuralNetwork import NeuralNetwork


GRID_COUNT = 3
ROW_LENGTH = 3
COLUMN_LENGTH = 3

# 27 inputs per player, 1 for score difference, and 1 for game progress
INPUT_SIZE = GRID_COUNT * ROW_LENGTH * COLUMN_LENGTH * 2 + 1 + 1
OUTPUT_SIZE = GRID_COUNT * ROW_LENGTH * COLUMN_LENGTH


class NNPlayer:
    @staticmethod
    def getSpecs():
        return (INPUT_SIZE, OUTPUT_SIZE)

    def __init__(self, weights, biases, activiation_functions):
        self.neural_network = NeuralNetwork(
            weights,
            biases,
            activiation_functions
        )

    def play(self, player_state, opponent_state,
             player_score, opponent_score,
             turn, length, pip_count):

        flattened_player_state = numpy.array(player_state).flatten()
        flattened_opponent_state = numpy.array(opponent_state).flatten()

        board_state = numpy.concatenate(
            (flattened_player_state, flattened_opponent_state)
        )

        score_difference = (player_score - opponent_score) / (pip_count * turn)
        game_progress = turn / length
        match_state = numpy.array([score_difference, game_progress])

        # Construct neural network input array
        nn_inputs = numpy.concatenate((board_state, match_state))

        # Run neural network for current turn
        output = self.neural_network.propagate(nn_inputs)

        # Index on board as if a 1D array
        board_position = int(numpy.argmax(output))

        # Convert 1D index to grid, row, and column indexes
        grid = board_position % GRID_COUNT
        row = (board_position // GRID_COUNT) % ROW_LENGTH
        column = board_position // (ROW_LENGTH * COLUMN_LENGTH)
        
        move = [grid, row, column]

        return move

    def getNN(self):
        return self.neural_network
