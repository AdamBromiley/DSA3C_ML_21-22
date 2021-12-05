import unittest
import sys
import numpy as np
sys.path.append("../src")
# from sample import Layer
from Layer import Layer

identity = np.vectorize(lambda x:x)

class TestLayer(unittest.TestCase):
    def test_forward(self):
        M = np.ones([4,3])
        B = 2*np.ones(4)
        idL = Layer(M, B, identity)
        input = np.array([0, 0, 0])
        output = idL.forward(input)
        self.assertTrue( (output == np.array([2.0, 2.0, 2.0, 2.0])).all() )
        input = np.array([1, 0, 0])
        output = idL.forward(input)
        self.assertTrue( (output == np.array([3.0, 3.0, 3.0, 3.0])).all() )
        input = np.array([1, 1, 0])
        output = idL.forward(input)
        self.assertTrue( (output == np.array([4.0, 4.0, 4.0, 4.0])).all() )
        input = np.array([1, 1, 1])
        output = idL.forward(input)
        self.assertTrue( (output == np.array([5.0, 5.0, 5.0, 5.0])).all() )

    def test_constructor_setting(self):
        M = np.ones([4,3])
        B = 2*np.ones(4)
        idL = Layer(M, B, identity)
        self.assertTrue((idL.getMatrix() == M).all())
        self.assertTrue((idL.getBiasVector() == B).all())
        self.assertTrue(idL.getFunction() == identity)
    
    def test_constructor_copying(self):
        M = np.ones([4,3])
        B = 2*np.ones(4)
        idL = Layer(M, B, identity)
        M[0][0] = 10.0
        B[0] = 10.0
        self.assertTrue(not (idL.getMatrix() == M).all())
        self.assertTrue(not (idL.getBiasVector() == B).all())
    
    def test_constructor_exception(self):
        M = np.ones([3,4])
        B = np.ones(5)
        self.assertRaises(ValueError , Layer, M, B, identity)

if __name__ == '__main__':
    unittest.main()