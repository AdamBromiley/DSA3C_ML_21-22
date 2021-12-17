import copy
from multiprocessing import Pool
import os
import pickle
import random
import signal
import statistics
import sys

import numpy

from CompromiseGame import CompromiseGame, RandomPlayer, GreedyPlayer, SmartGreedyPlayer
from Layer import linear, relu
from NNPlayer import NNPlayer, INPUT_SIZE, OUTPUT_SIZE


# Number of players in the population
POPULATION_SIZE = 200

# Number of generations
GENERATIONS = 4000

# Number of games each player plays against each opponent in a generation
GAMES_PLAYED = 20


# Player fitness is determined by recording their win count; each win
# contributes 1 to this score. How much the player wins by in a given game can
# contribute to the fitness by setting this to a non-zero value.
SCORE_DIFFERENCE_WEIGHTING = 0.5


# Number of points used in the k-point crossover
CROSSOVER_POINTS = 2

# Percentage of population to be selected as parents for breeding. This changes
# linearly over the training period between the start and end value.
CROSSOVER_CHANCE_START = 0.8
CROSSOVER_CHANCE_END = 0.8

# Chance that a gene (weight or bias) is mutated. This changes linearly over
# the training period between the start and end value.
MUTATION_CHANCE_START = 0.4
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

# Pickle dump of a population to load (optional)
POPULATION_LOAD_FILEPATH = ""


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

        weights.append(rng.normal(0, 2 / inputs, (outputs, inputs)))
        biases.append(numpy.full(outputs, 0.01))

    activation_functions = [relu, relu, linear]

    player = NNPlayer(weights, biases, activation_functions)
    player.win_count = 0
    return player


def play_game(player):
    player.win_count = 0
    player.fitness = 0

    for opponent in opponents:
        compromise_game = CompromiseGame(player, opponent, 30, 10)

        for _ in range(GAMES_PLAYED):
            score = compromise_game.play()
            compromise_game.resetGame()

            if score[0] > score[1]:
                player.win_count += 1
                player.fitness += opponent.weighting * (
                    1 + \
                    SCORE_DIFFERENCE_WEIGHTING * \
                    (score[0] - score[1]) / (score[0] + score[1])
                )

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
        child1, child2 = breed(
            breedable_population[i],
            breedable_population[i + 1],
            mutation_rate
        )
        offspring += [child1, child2]

    return offspring


def breed(parent1, parent2, mutation_rate):
    # We deepcopy so the original parents remain unmodified
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    crossover(child1, child2)
    mutate(child1, mutation_rate)
    mutate(child2, mutation_rate)

    return child1, child2


def crossover(parent1, parent2):
    parent1_layers = parent1.getNN().getLayers()
    parent2_layers = parent2.getNN().getLayers()

    for parent1_layer, parent2_layer in zip(parent1_layers, parent2_layers):
        parent1_biases = parent1_layer.getBiasVector()
        parent2_biases = parent2_layer.getBiasVector()

        # Get array of random crossover points
        crossover_points = random.sample(
            range(len(parent1_biases)), CROSSOVER_POINTS
        )

        parent1_layer.biases, parent2_layer.biases = k_point_crossover(
            parent1_biases,
            parent2_biases,
            crossover_points
        )


        parent1_weights = parent1_layer.getMatrix()
        parent2_weights = parent2_layer.getMatrix()

        flattened_parent1_weights = numpy.ndarray.flatten(parent1_weights)
        flattened_parent2_weights = numpy.ndarray.flatten(parent2_weights)

        # Get array of random crossover points
        crossover_points = random.sample(
            range(len(flattened_parent1_weights)), CROSSOVER_POINTS
        )

        new_flattened_parent1_weights, new_flattened_parent2_weights = k_point_crossover(
            flattened_parent1_weights,
            flattened_parent2_weights,
            crossover_points
        )

        parent1_layer.weights = new_flattened_parent1_weights.reshape(parent1_weights.shape)
        parent2_layer.weights = new_flattened_parent2_weights.reshape(parent2_weights.shape)


def k_point_crossover(array1, array2, crossover_points):
    new_array1 = array1
    new_array2 = array2

    for point in crossover_points:
        new_array1, new_array2 = single_point_crossover(new_array1, new_array2, point)

    return new_array1, new_array2


def single_point_crossover(array1, array2, crossover_point):
    new_array1 = numpy.append(
        array1[:crossover_point],
        array2[crossover_point:]
    )

    new_array2 = numpy.append(
        array2[:crossover_point],
        array1[crossover_point:]
    )

    return new_array1, new_array2


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

        mask = numpy.random.choice(
            [0, 1],
            weights.shape,
            p=[1 - mutation_rate, mutation_rate]
        ).astype(bool)

        random_neuron = rng.normal(0, 0.05, weights.shape)
        weights[mask] += random_neuron[mask]


structure = [INPUT_SIZE, 22, 24, OUTPUT_SIZE]
layer_count = len(structure)


if len(POPULATION_LOAD_FILEPATH) == 0:
    players = []

    for i in range(POPULATION_SIZE):
        players.append(create_player())
else:
    with open(POPULATION_LOAD_FILEPATH, "rb") as f:
        players = pickle.load(f)


random_opponent = RandomPlayer()
greedy_opponent = GreedyPlayer()
smart_greedy_opponent = SmartGreedyPlayer()

random_opponent.weighting = 1
greedy_opponent.weighting = 1.25
smart_greedy_opponent.weighting = 1.5

opponents = [random_opponent, greedy_opponent, smart_greedy_opponent]
total_games_played = GAMES_PLAYED * len(opponents)

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

            worst_win_rate = min(win_counts) / total_games_played
            lower_quartile = numpy.quantile(win_counts, 0.25) / total_games_played
            mean_win_rate = statistics.mean(win_counts) / total_games_played
            median_win_rate = statistics.median(win_counts) / total_games_played
            upper_quartile = numpy.quantile(win_counts, 0.75) / total_games_played
            best_win_rate = max(win_counts) / total_games_played

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
