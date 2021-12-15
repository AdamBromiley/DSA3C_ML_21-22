import copy
from multiprocessing import Pool
import os
import pickle
import random
import signal
import statistics
import sys

import numpy

from CompromiseGame_optimised import CompromiseGame, RandomPlayer
from Layer import linear, relu, tanh
from NNPlayer import NNPlayer, INPUT_SIZE, OUTPUT_SIZE


# Number of players in the population
POPULATION_SIZE = 1000

# Number of generations
GENERATIONS = 8000

# Number of games each player plays in a generation
GAMES_PLAYED = 40


# Player fitness is determined by recording their win count; each win
# contributes 1 to this score. How much the player wins by in a given game can
# contribute to the fitness by setting this to a non-zero value.
SCORE_DIFFERENCE_WEIGHTING = 0.5


# 51.13.108.128
# 20.203.186.64
# 20.79.222.21
# Best = 400, 1000, 0.3-0.8, 0.3-0.005
# Number of points used in the k-point crossover
CROSSOVER_POINTS = 2

# Percentage of population to be selected as parents for breeding. This changes
# linearly over the training period between the start and end value.
CROSSOVER_CHANCE_START = 0.2
CROSSOVER_CHANCE_END = 0.8

# Chance that a gene (weight or bias) is mutated. This changes linearly over
# the training period between the start and end value.
MUTATION_CHANCE_START = 0.3
MUTATION_CHANCE_END = 0.005

# String used to name the population dump and results file
GENERATION_STRING = f"{POPULATION_SIZE}_{GENERATIONS}_{GAMES_PLAYED}"
GENERATION_STRING += f"_{CROSSOVER_CHANCE_START}_{CROSSOVER_CHANCE_END}"
GENERATION_STRING += f"_{MUTATION_CHANCE_START}_{MUTATION_CHANCE_END}"
GENERATION_STRING += f"_{CROSSOVER_POINTS}"

# Pickle dump of the final population
POPULATION_DUMP_FILEPATH = f"../populations/{GENERATION_STRING}.dat"

# CSV file storing the win-rate for every generation
RESULTS_FILEPATH = f"../results/{GENERATION_STRING}.csv"


# Create numpy RNG generator
rng = numpy.random.default_rng()


def create_unique_filename(filepath):
    filename, file_extension = os.path.splitext(filepath)
    file_existence_count = 1

    while os.path.exists(filepath):
        filepath = f"{filename} ({file_existence_count}){file_extension}"
        file_existence_count += 1

    return filepath


def save_population():
    with open(population_dump_filepath, "wb") as f:
        pickle.dump(players, f)


def signal_handler(signal, stack_frame):
    save_population()
    sys.exit(0)


def create_player():
    weights = []
    biases = []
    activation_functions = []

    for i in range(layer_count - 1):
        inputs = structure[i]
        outputs = structure[i + 1]

        weights.append(
                rng.standard_normal(
                    (outputs, inputs)
                ) * numpy.sqrt(2 / (outputs - 1)
            )
        )
        biases.append(numpy.full(outputs, 0.01))

    activation_functions = [relu, relu, linear]

    player = NNPlayer(weights, biases, activation_functions)
    player.win_count = 0
    return player


def play_game(player):
    player.win_count = 0
    player.fitness = 0

    compromise_game = CompromiseGame(player, opponent, 30, 10)

    for _ in range(GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            player.win_count += 1
            player.fitness += 1 + \
                SCORE_DIFFERENCE_WEIGHTING * \
                (score[0] - score[1]) / (score[0] + score[1])

    return player


def breed_population(population, crossover_rate, mutation_rate):
    population_size = len(population)

    ranks = numpy.arange(population_size - 1, -1, -1)
    weights = numpy.exp(ranks / population_size)
    weights /= weights.sum()

    breedable_population = numpy.random.choice(
        population,
        int(population_size * crossover_rate),
        p=weights
    )

    offspring = []

    for i in range(0, len(breedable_population) - 1, 2):
        child1 = copy.deepcopy(breedable_population[i])
        child2 = copy.deepcopy(breedable_population[i + 1])
        breed(child1, child2, mutation_rate)
        offspring += [child1, child2]

    return offspring


def breed(daddy, mummy, mutation_rate):
    crossover(daddy, mummy)
    mutate(daddy, mutation_rate)
    mutate(mummy, mutation_rate)


def crossover(daddy, mummy):
    daddy_layers = daddy.getNN().getLayers()
    mummy_layers = mummy.getNN().getLayers()

    for daddy_layer, mummy_layer in zip(daddy_layers, mummy_layers):
        daddy_biases = daddy_layer.getBiasVector()
        mummy_biases = mummy_layer.getBiasVector()

        # Get array of random crossover points
        crossover_points = random.sample(
            range(len(daddy_biases)), CROSSOVER_POINTS
        )

        k_point_crossover(daddy_biases, mummy_biases, crossover_points)

        daddy_weights = daddy_layer.getMatrix()
        mummy_weights = mummy_layer.getMatrix()

        flattened_daddy_weights = numpy.ndarray.flatten(daddy_weights)
        flattened_mummy_weights = numpy.ndarray.flatten(mummy_weights)

        # Get array of random crossover points
        crossover_points = random.sample(
            range(len(flattened_daddy_weights)), CROSSOVER_POINTS
        )

        k_point_crossover(
            flattened_daddy_weights,
            flattened_mummy_weights,
            crossover_points
        )

        daddy_weights = flattened_daddy_weights.reshape(daddy_weights.shape)
        mummy_weights = flattened_mummy_weights.reshape(mummy_weights.shape)


def k_point_crossover(array1, array2, crossover_points):
    for point in crossover_points:
        single_point_crossover(array1, array2, point)


def single_point_crossover(array1, array2, crossover_point):
    new_array1 = numpy.append(
        array1[:crossover_point],
        array2[crossover_point:]
    )

    new_array2 = numpy.append(
        array2[:crossover_point],
        array1[crossover_point:]
    )

    array1 = new_array1
    array2 = new_array2


def mutate(child, mutation_rate):
    for layer in child.getNN().layers:
        biases = layer.getBiasVector()
        weights = layer.getMatrix()

        mask = numpy.random.choice(
            [0, 1],
            biases.shape,
            p=[1 - mutation_rate, mutation_rate]
        ).astype(bool)

        random_biases = rng.uniform(-0.05, 0.05, biases.shape)
        biases[mask] += random_biases[mask]

        # for neuron in weights:
        #     mask = numpy.random.choice(
        #         [0, 1],
        #         neuron.shape,
        #         p=[1 - mutation_rate, mutation_rate]
        #     ).astype(bool)

        #     random_neuron = rng.normal(0, 0.05, neuron.shape)
        #     neuron[mask] += random_neuron[mask]

        mask = numpy.random.choice(
            [0, 1],
            weights.shape,
            p=[1 - mutation_rate, mutation_rate]
        ).astype(bool)

        random_neuron = rng.normal(0, 0.05, weights.shape)
        weights[mask] += random_neuron[mask]


structure = [INPUT_SIZE, 22, 24, OUTPUT_SIZE]
layer_count = len(structure)


players = []

for i in range(POPULATION_SIZE):
    players.append(create_player())


opponent = RandomPlayer()

results_filepath = create_unique_filename(RESULTS_FILEPATH)
population_dump_filepath = create_unique_filename(POPULATION_DUMP_FILEPATH)

signal.signal(signal.SIGINT, signal_handler)

with open(results_filepath, "w") as f:
    results_header = f"Generation, Worst, Lower Quartile, Mean, Median"
    results_header += f", Upper Quartile, Best, Median Fitness"
    f.write(results_header + "\n")
    f.flush()

    with Pool() as pool:
        for i in range(GENERATIONS):
            players = pool.map(play_game, players)

            # Sort players array by player.fitness (descending)
            players.sort(key=lambda player: player.fitness, reverse=True)


            # Get win-rate statistics
            win_counts = [player.win_count for player in players]
            worst_win_rate = min(win_counts) / GAMES_PLAYED
            lower_quartile = numpy.quantile(win_counts, 0.25) / GAMES_PLAYED
            mean_win_rate = statistics.mean(win_counts) / GAMES_PLAYED
            median_win_rate = statistics.median(win_counts) / GAMES_PLAYED
            upper_quartile = numpy.quantile(win_counts, 0.75) / GAMES_PLAYED
            best_win_rate = max(win_counts) / GAMES_PLAYED

            # Get fitness statistics
            fitnesses = [player.fitness for player in players]
            median_fitness = statistics.median(fitnesses)

            s = f"{i}, {worst_win_rate}, {lower_quartile}, {mean_win_rate}"
            s += f", {median_win_rate}, {upper_quartile}, {best_win_rate}"
            s += f", {median_fitness}"
            f.write(s + "\n")
            f.flush()

            print(s)

            # Dynamic decreasing of high mutation ratio/dynamic increasing of
            # low crossover ratio (DHM/ILC)
            crossover_rate = CROSSOVER_CHANCE_START + i * \
                (CROSSOVER_CHANCE_END - CROSSOVER_CHANCE_START) / GENERATIONS
            
            mutation_rate = MUTATION_CHANCE_START + i * \
                (MUTATION_CHANCE_END - MUTATION_CHANCE_START) / GENERATIONS

            offspring = breed_population(
                players, crossover_rate, mutation_rate
            )

            ranks = numpy.arange(POPULATION_SIZE - 1, -1, -1)
            weights = numpy.exp(ranks / POPULATION_SIZE)
            weights /= weights.sum()

            breedable_population = numpy.random.choice(
                players,
                int(POPULATION_SIZE * crossover_rate),
                p=weights
            )

            players = list(
                numpy.random.choice(players, len(players) - len(offspring))
            ) + offspring

save_population()
