
import numpy as np

from src.preprocessing import linear_cuts

class Test_linear_cuts(object):
    
    def test_one_equation(self):
        
        xs = np.array([[0, 0],
                       [1, 1],
                       [-1, -1]])
        equations = np.array([[1, -1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        np.testing.assert_array_equal(sets, np.array([[1, 1, 1]], dtype=bool))
                
        xs = np.array([[0, 0],
                       [0, 1],
                       [0, -1]])
        equations = np.array([[1, -1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        np.testing.assert_array_equal(sets, np.array([[1, 0, 1]], dtype=bool))
        
        xs = np.array([[0, 0],
                       [1, 1],
                       [-1, -1]])
        equations = np.array([[-1, 1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        np.testing.assert_array_equal(sets, np.array([[1, 1, 1]], dtype=bool))        
        
        xs = np.array([[0, 0],
                       [0, 1],
                       [0, -1]])
        
        equations = np.array([[-1, 1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        np.testing.assert_array_equal(sets, np.array([[1, 0, 1]], dtype=bool))

        
    def test_two_equation(self):
        
        xs = np.array([[0, 0],
                       [0, 1],
                       [0, -1]])
        
        equations = np.array([[-1, 1, 0], [1, -1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        
        np.testing.assert_array_equal(np.unique(sets, axis=0), np.array([[1, 0, 1]], dtype=bool))
        
    def test_four_equation(self):
        
        xs = np.array([[0, 0],
                       [0, 1],
                       [0, -1]])
        
        equations = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [1, 1, 0],
                              [1, -1, 0]])
        
        sets, equations = linear_cuts(xs, equations)
        
        np.testing.assert_array_equal(sets, np.array([[1, 1, 1],
                                                      [1, 1, 0],
                                                      [1, 1, 0],
                                                      [1, 0, 1]], dtype=bool))
