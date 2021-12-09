import curses
import random
import re
import math

import numpy


probabilities = numpy.full(27, 1 / 27)


class AbstractPlayer:
    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return numpy.random.randint(0, 2, 3)

    def placePips(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return numpy.random.randint(0, 2, (nPips, 3))


class RandomPlayer(AbstractPlayer):
    pass


class CompromiseGame:
    def __init__(self, playerA, playerB, nPips, length, gametype = "s", noTies = True):
        self.noties = noTies
        self.gameLength = length
        self.turn = 0
        self.redPlayer = playerA
        self.greenPlayer = playerB
        self.newPips = nPips
        self.greenScore = 0
        self.redScore = 0
        self.greenPips = numpy.zeros((3, 3, 3), dtype=int)
        self.redPips = numpy.zeros((3, 3, 3), dtype=int)

    def resetGame(self):
        self.turn = 0
        self.greenScore = 0
        self.redScore = 0
        self.greenPips = numpy.zeros((3, 3, 3), dtype=int)
        self.redPips = numpy.zeros((3, 3, 3), dtype=int)

    def roundStart(self):
        self.turn += 1
        self.redPips += numpy.random.multinomial(self.newPips, probabilities).reshape(3, 3, 3)
        self.greenPips += numpy.random.multinomial(self.newPips, probabilities).reshape(3, 3, 3)

    def getMoves(self):
        self.redMove = self.redPlayer.play(self.redPips,self.greenPips, self.redScore, self.greenScore, self.turn, self.gameLength, self.newPips)
        self.greenMove = self.greenPlayer.play(None,None, self.greenScore, self.redScore, self.turn, self.gameLength, self.newPips)

    def updateScore(self):
        moves = list(zip(self.redMove, self.greenMove))
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    if not (i in moves[0] or j in moves[1] or k in moves[2]):
                        self.redScore += self.redPips[i][j][k]
                        self.greenScore += self.greenPips[i][j][k]
                        self.redPips[i][j][k] = 0
                        self.greenPips[i][j][k] = 0

    def playRound(self):
        self.roundStart()
        self.getMoves()
        self.updateScore()

    def play(self):
        while self.turn < self.gameLength or (self.noties and self.redScore == self.greenScore):
            self.playRound()
        return [self.redScore, self.greenScore]
