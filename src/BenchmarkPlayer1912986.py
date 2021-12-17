from statistics import mean
from Player1912986 import NNPlayer
from CompromiseGame import CompromiseGame, GreedyPlayer, RandomPlayer, SmartGreedyPlayer

GAMES_PLAYED = 400

player = NNPlayer()

opponents = {
    "random": RandomPlayer(),
    "greedy": GreedyPlayer(),
    "smart_greedy": SmartGreedyPlayer()
}

win_rates = []

for opponent_name, opponent in opponents.items():
    win_count = 0

    compromise_game = CompromiseGame(player, opponent, 30, 10)

    for game in range(GAMES_PLAYED):
        score = compromise_game.play()
        compromise_game.resetGame()

        if score[0] > score[1]:
            win_count += 1

    win_rate = win_count / GAMES_PLAYED
    win_rates.append(win_rate)
    
    setattr(player, f"{opponent_name}_win_rate", win_rate)
    print(f"{opponent_name} = f{win_rate}")

print(f"mean = {mean(win_rates)}")