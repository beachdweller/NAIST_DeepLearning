import unittest
import lecture1_code00 as dl
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


class TestDeepLearning(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    def tearDown(self):
        del self.X
        del self.Y

    def test_linear_model_00(self):
        x = [1.0, -1.0]
        w = [1, 1, 0]
        result = dl.linear_model(w, x)

        self.assertAlmostEqual(x[0]*w[0] + x[1]*w[1] + w[2] * 1, result,)

        x3 = [1.0, -1.0, 1.0]
        w3 = [1, 2, 1, 0.5]
        result3 = dl.linear_model(w3, x3)

        self.assertAlmostEqual(x3[0]*w3[0] + x3[1]*w3[1] + x3[2] * w3[2] + w3[3] * 1.0, result3,)

    def test_stochastic_gradient_descent(self):
        w_list = dl.stochastic_gradient_descent(self.X, self.Y, 1, heuristic=False)
        w_mine = w_list[-1]

        self.assertLess(dl.loss_function(w_mine, self.X, self.Y), 2.0)

        expected = (1.45057928e+000,  -7.90299460e-001,  -6.66694597e-106)
        for w, e in zip(w_mine, expected):
            self.assertAlmostEqual(w, e)


    def test_get_sample_format(self):
        X = np.zeros((100, 100))
        self.assertEqual('%03d', dl.get_sample_format(X))
