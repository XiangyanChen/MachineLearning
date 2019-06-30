"""
This code was done by FlameCharmander
which is published in https://github.com/FlameCharmander/MachineLearning
and my csdn blog is https://blog.csdn.net/tudaodiaozhale
everyone is welcome to contact me via 13030880@qq.com
"""

import numpy as np

class PerceptronDual:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sign(self, x):  # sign function
        return 1 if x >= 0 else -1

    def train(self, data_set, labels):
        lr = 1
        n = np.array(data_set).shape[0] #n means rows
        data_set = np.mat(data_set)
        alpha = np.zeros(n) #alpha means the total iteration of wrong point
        bias = 0
        i = 0
        while i < n:
            #in this step, we elide gram matrix
            if (labels[i] * self.sign(sum(alpha * labels * data_set * data_set[i].T)+bias) == -1):
                alpha[i] = alpha[i] + lr
                bias = bias + lr * labels[i]
                i = 0
            else:
                i += 1
        self.weights = sum(alpha * labels * data_set)
        self.bias = bias

    def predict(self, data):
        data = np.array([data])
        if (self.weights is not None and self.bias is not None):
            return self.sign((self.weights * data.T) + self.bias)
        else:
            return 0

if __name__ == '__main__':
    """
            this code is corresponding to algorithm(2.1) in P29
        """
    data_set = [[3, 3],
                [4, 3],
                [1, 1]]
    labels = [1, 1, -1]
    perceptron = PerceptronDual()
    perceptron.train(data_set, labels)
    print(perceptron.weights)
    result = perceptron.predict([1, 1])
    print(result)