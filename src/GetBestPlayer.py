from multiprocessing import Pool
import numpy
from operator import attrgetter
import pickle

from CompromiseGame import CompromiseGame, RandomPlayer, GreedyPlayer, SmartGreedyPlayer


# Number of games each player plays in the competition to determine the best
# player.
COMPETITION_GAMES_PLAYED = 200

# Pickle dump of the final population
POPULATION_DUMP_FILEPATH = f"../populations/200_4000_20_0.8_0.8_0.4_0.005_2.dat"


def play_game(player):
    player.random_win_count = 0
    player.greedy_win_count = 0
    player.smart_greedy_win_count = 0

    compromise_game = CompromiseGame(player, random_opponent, 30, 10)

    for _ in range(COMPETITION_GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            player.random_win_count += 1

    compromise_game = CompromiseGame(player, greedy_opponent, 30, 10)

    for _ in range(COMPETITION_GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            player.greedy_win_count += 1

    compromise_game = CompromiseGame(player, smart_greedy_opponent, 30, 10)

    for _ in range(COMPETITION_GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            player.smart_greedy_win_count += 1

    player.win_count = player.random_win_count + \
        player.greedy_win_count + \
        player.smart_greedy_win_count

    return player


with open(POPULATION_DUMP_FILEPATH, "rb") as f:
    players = pickle.load(f)


random_opponent = RandomPlayer()
greedy_opponent = GreedyPlayer()
smart_greedy_opponent = SmartGreedyPlayer()

with Pool() as pool:
    players = pool.map(play_game, players)

with open("best_players.txt", "w") as f:
    best_random_player = max(players, key=attrgetter("random_win_count"))
    f.write(f"Best random win-rate: {best_random_player.random_win_count / 200} (g: {best_random_player.greedy_win_count / 200}, s: {best_random_player.smart_greedy_win_count / 200})\n")

    best_greedy_player = max(players, key=attrgetter("greedy_win_count"))
    f.write(f"Best greedy win-rate: {best_greedy_player.greedy_win_count / 200} (r: {best_greedy_player.random_win_count / 200}, s: {best_greedy_player.smart_greedy_win_count / 200})\n")

    best_smart_greedy_player = max(players, key=attrgetter("smart_greedy_win_count"))
    f.write(f"Best smart greedy win-rate: {best_smart_greedy_player.smart_greedy_win_count / 200} (r: {best_smart_greedy_player.random_win_count / 200}, g: {best_smart_greedy_player.greedy_win_count / 200})\n")

    best_player = max(players, key=attrgetter("win_count"))
    f.write(f"Best overall win-rate: {best_player.win_count / 600} (r: {best_player.random_win_count / 200} g: {best_random_player.greedy_win_count / 200}, s: {best_random_player.smart_greedy_win_count / 200})\n")


    numpy.set_printoptions(precision=None)

    for i, player in enumerate([best_random_player, best_greedy_player, best_smart_greedy_player, best_player]):
        f.write(f"Player {i}:\n")

        player_layers = player.getNN().getLayers()

        comma=","
        f.write("weights = [\n")
        for i, layer in enumerate(player_layers):
            if i == len(player_layers) - 1:
                comma=""

            weights = layer.getMatrix()
            f.write(f"{numpy.array2string(weights, separator=',')}{comma}\n")
        f.write("]\n")

        comma=","
        f.write("biases = [\n")
        for i, layer in enumerate(player_layers):
            if i == len(player_layers) - 1:
                comma=""

            biases = layer.getBiasVector()
            f.write(f"{numpy.array2string(biases, separator=',')}{comma}\n")
        f.write("]\n")
