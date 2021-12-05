import unittest
import sys
import numpy as np
sys.path.append("../src")
# from sample import Layer
# from sample import NeuralNetwork
from Layer import Layer
from NeuralNetwork import NeuralNetwork

identity = np.vectorize(lambda x:x)
square = np.sqrt

class TestLayer(unittest.TestCase):
    def test_constructor(self):
        M1 = np.ones([4,3])
        M2 = np.ones([5,4])
        M3 = np.ones([3,5])
        B1 = np.ones(4)
        B2 = np.ones(5)
        B3 = np.ones(3)
        Ms = [M1, M2, M3]
        Bs = [B1, B2, B3]
        Fs = [identity, identity, identity]
        nn = NeuralNetwork(Ms, Bs, Fs)
        Ls = nn.getLayers()
        for i in range(len(Ls)):
            self.assertTrue(type(Ls[i]) == Layer)
            self.assertTrue( (Ls[i].getMatrix() == Ms[i]).all() )
            self.assertTrue( (Ls[i].getBiasVector() == Bs[i]).all() )
            self.assertTrue( Ls[i].getFunction() == Fs[i] )

    def test_constructor_errors(self):
        M1 = np.ones([4,3])
        M2 = np.ones([5,4])
        M3 = np.ones([3,5])
        B1 = np.ones(4)
        B2 = np.ones(5)
        B3 = np.ones(3)
        Ms = [M1, M2, M3]
        Bs = [B1, B2, B3]
        Fs = [identity, identity]
        self.assertRaises(ValueError , NeuralNetwork, Ms, Bs, Fs) # not enough functions
        Fs = [identity, identity, identity]
        Bs = [B1, B2]
        self.assertRaises(ValueError , NeuralNetwork, Ms, Bs, Fs) # not enough biases
        Bs = [B1, B2, B3]
        Ms = [M1, M2]
        self.assertRaises(ValueError , NeuralNetwork, Ms, Bs, Fs) # not enough matrices
        Ms = [M1, M2, M3]
        M2 = np.ones([4,2])
        Ms = [M1, M2, M3]
        self.assertRaises(ValueError , NeuralNetwork, Ms, Bs, Fs) # mismatch matrix sizes
        
    def test_run(self):
        M1 = np.ones([4,3])
        M2 = np.ones([5,4])
        M3 = np.ones([3,5])
        B1 = np.ones(4)
        B2 = np.ones(5)
        B3 = np.ones(3)
        Ms = [M1, M2, M3]
        Bs = [B1, B2, B3]
        Fs = [square, square, square]
        nn = NeuralNetwork(Ms, Bs, Fs)
        input = np.ones(3)
        self.assertTrue( (nn.propagate(input) == 4*np.ones(3)).all() )

if __name__ == '__main__':
    unittest.main()