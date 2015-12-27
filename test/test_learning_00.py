import unittest
import lecture1_code00 as dl
from sklearn.datasets.samples_generator import make_blobs


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
