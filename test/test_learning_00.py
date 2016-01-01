import unittest
import lecture1_code00 as dl
from sklearn.datasets.samples_generator import make_blobs
import numpy as np


class TestLearning(unittest.TestCase):
    def setUp(self):
        self.X, self.Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    def tearDown(self):
        del self.X
        del self.Y

    def test_stochastic_gradient_descent(self):
        w_list = dl.stochastic_gradient_descent(self.X, self.Y, 1, heuristic=False)
        w_mine = w_list[-1]

        self.assertLess(dl.loss_function(w_mine, self.X, self.Y), 2.0)

        expected = (1.45057928e+000,  -7.90299460e-001,  -6.66694597e-106)
        for w, e in zip(w_mine, expected):
            self.assertAlmostEqual(w, e)

    def test_loss_function(self):
        w = (1.45057928e+000,  -7.90299460e-001,  0)
        result = dl.loss_function(w, self.X, self.Y)
        expected = 0.9916009776743884

        self.assertAlmostEqual(expected, result)

    def test_gradient_descent_n(self):
        w_list = dl.gradient_descent_n(self.X, self.Y, 1.0, 20)
        expected = (8.478566520916212, -8.248713639079872, 3.014805375759964)
        for w, e in zip(w_list[-1], expected):
            self.assertAlmostEqual(w, e)

    def test_gradient_descent_step(self):
        w_initial = dl.init_w([], self.X)
        loss_initial = dl.loss_function(w_initial, self.X, self.Y)

        w_next = dl.gradient_descent_step(self.X, self.Y, 1, w_initial)
        loss_next = dl.loss_function(w_next, self.X, self.Y)

        self.assertGreater(loss_initial, loss_next)

    def test_init_w(self):
        self.assertSequenceEqual([1.0, 1.0, 1.0], dl.init_w([], self.X).tolist())

    def test_init_filename_format_string(self):
        self.assertEqual('abc%02d.png', dl.init_filename_format_string('abc', self.X))
        self.assertIsNone(dl.init_filename_format_string(None, self.X))

    def test_two_layer_neural_net(self):
        w1 = np.array(np.ones((3, 4)))
        w2 = [1.0, 1.0, 1.0, 1.0, ]
        result = dl.two_layer_neural_net(w1, w2, self.X)
        expected = 0.0
        self.assertEqual((self.X.shape[0], 1), result.shape)

    def test_get_sample_format(self):
        self.assertEqual('%02d', dl.get_sample_format(self.X))


class TestLearningComponents(unittest.TestCase):
    def test_linear_model(self):
        w1 = [1.0, 1.0]
        x1 = [2.0]
        result1 = dl.linear_model(w1, x1)
        expected1 = 3.0
        self.assertEqual(expected1, result1)

        w2 = [1.0, -1.0]
        x2 = [2.0]
        result2 = dl.linear_model(w2, x2)
        expected2 = 1.0
        self.assertEqual(expected2, result2)

    def test_linear_model_00(self):
        x = [1.0, -1.0]
        w = [1, 1, 0]
        result = dl.linear_model(w, x)

        self.assertAlmostEqual(x[0]*w[0] + x[1]*w[1] + w[2] * 1, result,)

        x3 = [1.0, -1.0, 1.0]
        w3 = [1, 2, 1, 0.5]
        result3 = dl.linear_model(w3, x3)

        self.assertAlmostEqual(x3[0]*w3[0] + x3[1]*w3[1] + x3[2] * w3[2] + w3[3] * 1.0, result3,)

    def test_one_matrix(self):
        x = np.array([[0.0, 0.0],
                     [0.0, 0.0]])

        x1 = dl.append_one_matrix(x)

        self.assertSequenceEqual((2, 3), x1.shape)

        x1_list = x1.tolist()
        expected_list = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        self.assertSequenceEqual(expected_list, x1_list)
