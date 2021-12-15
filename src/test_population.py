from multiprocessing import Pool
import numpy
from operator import attrgetter
import pickle

from CompromiseGame_optimised import CompromiseGame, RandomPlayer


# Number of games each player plays in the competition to determine the best
# player.
COMPETITION_GAMES_PLAYED = 400

# Pickle dump of the final population
POPULATION_DUMP_FILEPATH = f"../populations/something.dat"


def play_game(player):
    player.win_count = 0

    compromise_game = CompromiseGame(player, opponent, 30, 10)

    for _ in range(COMPETITION_GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            player.win_count += 1

    return player


with open(POPULATION_DUMP_FILEPATH, "rb") as f:
    players = pickle.load(f)


opponent = RandomPlayer()

with Pool() as pool:
    players = pool.map(play_game, players)

best_player = max(players, key=attrgetter("win_count"))


numpy.set_printoptions(precision=None)

best_player_layers = best_player.getNN().getLayers()

comma=","
print("weights = [")
for i, layer in enumerate(best_player_layers):
    if i == len(best_player_layers) - 1:
        comma=""

    weights = layer.getMatrix()
    print(f"{numpy.array2string(weights, separator=',')}{comma}")
print("]")

comma=","
print("biases = [")
for i, layer in enumerate(best_player_layers):
    if i == len(best_player_layers) - 1:
        comma=""

    biases = layer.getBiasVector()
    print(f"{numpy.array2string(biases, separator=',')}{comma}")
print("]")
