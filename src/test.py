from Player1912986 import NNPlayer
from CompromiseGame import CompromiseGame, RandomPlayer

player = NNPlayer()
opponent = RandomPlayer()

game = CompromiseGame(player, opponent, 30, 10)

score = game.play()

print(score)