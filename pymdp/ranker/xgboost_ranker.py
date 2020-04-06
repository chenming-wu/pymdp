from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class XGBoost_Ranker():
    def __init__(self, timestamp, load=True):
        self.model = XGBClassifier()
        self.model.load_model(timestamp+'.file')
        self.factor = 1.0
    
    def set_factor(self, factor):
        self.factor = factor

    def rank_features(self, features):
        _features = np.copy(features)
        for f in _features:
            f[1] *= self.factor
            f[4] *= self.factor
            f[5] *= self.factor
        # return np.array([0, 1, 2, 3, 4])

        test_x = []
        for i in range(len(_features)):
            for j in range(len(_features)):
                if i == j:
                    continue
                test_x.append(np.concatenate(
                (_features[i], _features[j]), axis=0))
        
        test_x = np.array(test_x)
        print(test_x.shape)
        y = self.model.predict(test_x).reshape(len(_features), len(_features)-1)
        y = np.sum(y, axis=1)
        # print(y)
        return np.argsort(y)[::-1]