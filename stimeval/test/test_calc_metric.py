import unittest
from stimeval import StimEval
import numpy as np

import sys
sys.path.append('../')

class TestCalc(unittest.TestCase):

    def test_rms(self):
        #test rms return 0 for the same input
        rand_img = np.random.rand(1,224,224,3)
        eval = StimEval(metric='root_mean_square')
        self.assertAlmostEqual(eval(rand_img, rand_img), 0)