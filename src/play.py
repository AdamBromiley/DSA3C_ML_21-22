from CompromiseGame import CompromiseGame, RandomPlayer
from NNPlayer import NNPlayer


import math

import numpy

from Layer import relu

structure = [56, 64, 64, 27]
layer_count = len(structure)


weights = []
biases = []
activation_functions = []

for i in range(0, layer_count - 1):
    inputs = structure[i]
    outputs = structure[i + 1]

    weights.append(math.sqrt(2 / inputs) * numpy.random.randn(outputs, inputs))
    biases.append(numpy.full(outputs, 0.1))
    activation_functions.append(relu)

player = NNPlayer(weights, biases, activation_functions)

opponent = RandomPlayer()

game = CompromiseGame(player, opponent, 30, 10)

score = game.play()

print(f"NNPlayer got score {score[0]}, RandomPlayer got score {score[1]}")
