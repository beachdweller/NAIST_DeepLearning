import unittest
import lecture1_code00 as dl


class TestDeepLearning(unittest.TestCase):
    def test_linear_model_00(self):
        x = [1.0, -1.0]
        w = [1, 1, 0]
        result = dl.linear_model(w, x)

        self.assertAlmostEqual(x[0]*w[0] + x[1]*w[1] + w[2] * 1, result,)

