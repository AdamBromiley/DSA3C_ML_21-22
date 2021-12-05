import unittest
import sys
import numpy as np
import random
sys.path.append("../src")
# from sample import NNPlayer
# from sample import NeuralNetwork
from NNPlayer import NNPlayer
from NeuralNetwork import NeuralNetwork

identity = np.vectorize(lambda x:x)

class TestNNPlayer(unittest.TestCase):
    def test_getSpecs(self):
        specs = NNPlayer.getSpecs()
        self.assertTrue( len(specs) == 2 )
        for i in range(2):
            self.assertTrue( type(specs[i]) == int )
            self.assertTrue( specs[i] > 0 )

    def test_constructor(self):
        i,o = NNPlayer.getSpecs()
        M1 = np.ones([4,i])
        M2 = np.ones([5,4])
        M3 = np.ones([o,5])
        B1 = np.ones(4)
        B2 = np.ones(5)
        B3 = np.ones(o)
        Ms = [M1, M2, M3]
        Bs = [B1, B2, B3]
        Fs = [identity, identity, identity]
        pl = NNPlayer(Ms, Bs, Fs)
        nn = NeuralNetwork(Ms, Bs, Fs)
        pLs = pl.getNN().getLayers()
        nLs = nn.getLayers()
        for i in range(len(pLs)):
            self.assertTrue( (pLs[i].getMatrix() == nLs[i].getMatrix()).all() )
            self.assertTrue( (pLs[i].getBiasVector() == nLs[i].getBiasVector()).all() )
            self.assertTrue( pLs[i].getFunction() == nLs[i].getFunction() )
            
    def test_play(self):
        i,o = NNPlayer.getSpecs()
        M1 = 3*np.random.rand(4,i) - 1
        M2 = 3*np.random.rand(5,4) - 1
        M3 = 3*np.random.rand(o,5) - 1
        B1 = 3*np.random.rand(4) - 1
        B2 = 3*np.random.rand(5) - 1
        B3 = 3*np.random.rand(o) - 1
        Ms = [M1, M2, M3]
        Bs = [B1, B2, B3]
        Fs = [identity, identity, identity]
        pl = NNPlayer(Ms, Bs, Fs)
        for x in range(10):
            ownB = [[[random.randint(0, 5) for i in range(3)] for j in range(3)] for k in range(3)]
            oppB = [[[random.randint(0, 5) for i in range(3)] for j in range(3)] for k in range(3)]
            ownS = random.randint(0, 20)
            oppS = random.randint(0, 20)
            turn = random.randint(1, 10)
            gLen = random.randint(0, 10) + turn
            pips = random.randint(8, 20)
            move = pl.play(ownB, oppB, ownS, oppS, turn, gLen, pips)
            self.assertTrue( isinstance(move, list) )
            self.assertTrue( len(move) == 3 )
            for i in range(3):
                self.assertTrue( type(move[i]) == int )
                self.assertTrue( (move[i] <= 2) and (move[i] >= 0) )


if __name__ == '__main__':
    unittest.main()