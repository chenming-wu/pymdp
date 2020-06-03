import numpy as np
from .urank.evaluate_point import EvalPoint

class uRanker():
    def __init__(self, load=True):
        self.model = EvalPoint()
        self.factor = 1.0

    def set_factor(self, factor):
        self.factor = factor

    def rank_features(self, features):
        _features = np.copy(features)
        for f in _features:
            f[1] *= self.factor
            f[4] *= self.factor
            f[5] *= self.factor
            f[7] *= self.factor
            f[10] *= self.factor
            f[11] *= self.factor

        test_x = np.array(_features)
        test_x = test_x[:, 6::]
        print(test_x.shape)
        y = self.model.evaluate(test_x)
        return y